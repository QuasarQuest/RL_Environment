"""Training entry point — Hydra-configured, algorithm-agnostic.

Usage
-----
# Default run (configs/train.yaml + ppo/default.yaml + env/stage1.yaml):
    atb-train

# Override individual values:
    atb-train train.total_timesteps=5000000
    atb-train train.device=cpu

# Swap config groups:
    atb-train ppo=aggressive
    atb-train env=stage2
    atb-train ppo=aggressive env=stage2

# Compose a named experiment:
    atb-train +experiment=kl_sweep

# Multirun — grid search without Optuna:
    atb-train --multirun ppo.learning_rate=1e-4,3e-4 ppo.target_kl=0.01,0.02

# Resume from checkpoint:
    atb-train +train.resume=runs/run_s1_XYZ/checkpoints/step_100000
    atb-train +train.resume=runs/run_s1_XYZ/eval_best
"""
from __future__ import annotations

import os

# Disable HDF5's advisory file lock before anything opens a stats/trace .h5. With a
# read-only monitor polling the same file, the trainer's append-mode open would
# otherwise be refused with BlockingIOError (errno 11) and abort the run. Single
# writer + read-only monitors ⇒ safe. See monitoring/stats_writer.py for the rationale.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import math
import time
from pathlib import Path
from typing import cast

import hydra
import torch
import torch.nn as nn
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from rich.columns import Columns
from rich.console import Console
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecEnv, VecNormalize

from env.factory import build_vec_env
from network.export import export_to_onnx
from network.policy import ATB_POLICY_KWARGS
from training.algos import get_algo
from training.callbacks import (
    CheckpointCallback,
    EntropyCoefScheduleCallback,
    EpisodeStatsCallback,
    PeriodicPlotCallback,
    PolicyTelemetryCallback,
    RichLogCallback,
    kv_table,
    make_eval_callback,
)
from training.config import EnvConfig, PpoConfig, TrainConfig, register_configs
from training.schedules import linear_schedule
from training.smdp import SmdpDiscountCallback, smdp_buffer_class


def _seq_str(seq: nn.Sequential) -> str:
    parts = []
    for m in seq.children():
        if isinstance(m, nn.Conv2d):
            k = m.kernel_size[0]
            parts.append(f"Conv({m.in_channels}→{m.out_channels},{k}×{k})")
        elif isinstance(m, nn.Linear):
            parts.append(f"Linear({m.in_features}→{m.out_features})")
        elif isinstance(m, nn.LayerNorm):
            parts.append(f"LN({m.normalized_shape[0]})")
        elif isinstance(m, nn.Flatten):
            parts.append("Flatten")
    return " → ".join(parts)


def _print_arch(model) -> None:
    policy = model.policy
    fe = policy.features_extractor
    obs = tuple(policy.observation_space.shape)

    if hasattr(fe, "crop_cnn") and hasattr(fe, "mm_cnn"):
        console.print(f"  [dim]extractor[/dim]       : AtbCnnExtractor  obs={obs}")
        console.print(f"  [dim]  crop_cnn[/dim]      : {_seq_str(fe.crop_cnn)}")
        console.print(f"  [dim]  crop_head[/dim]     : {_seq_str(fe.crop_head)}")
        console.print(f"  [dim]  mm_cnn[/dim]        : {_seq_str(fe.mm_cnn)}")
        console.print(f"  [dim]  mm_head[/dim]       : {_seq_str(fe.mm_head)}")
        console.print(f"  [dim]  fusion[/dim]        : {_seq_str(fe.fusion)}")
    elif hasattr(fe, "net"):
        console.print(f"  [dim]extractor[/dim]       : MLP  obs={obs}  {_seq_str(fe.net)}")
    else:
        console.print(f"  [dim]extractor[/dim]       : {type(fe).__name__}  obs={obs}")

    if hasattr(policy, "mlp_extractor"):
        mlp = policy.mlp_extractor
        pi_mid = _seq_str(mlp.policy_net)
        vf_mid = _seq_str(mlp.value_net)
        pi_str = (f"{pi_mid} → " if pi_mid else "") + f"Linear(→{policy.action_net.out_features})"
        vf_str = (f"{vf_mid} → " if vf_mid else "") + "Linear(→1)"
        console.print(f"  [dim]pi[/dim]              : {pi_str}")
        console.print(f"  [dim]vf[/dim]              : {vf_str}")


def _sync_vecnorm_stats(
        src: VecNormalize | None,
        dst: VecNormalize | None,
) -> None:
    """Point dst's running stats at the same objects as src.

    Both obs_rms and ret_rms are RunningMeanStd instances. Assigning the
    reference (not a copy) means dst always reads the latest training stats
    without any explicit sync call — eval always sees what the policy sees.

    Called once after env construction and again after resume so the eval
    env is never left with freshly-initialised (mean=0, var=1) statistics.
    """
    if src is None or dst is None:
        return
    if src.norm_obs:
        dst.obs_rms = src.obs_rms
    dst.ret_rms = src.ret_rms


def _vecnorm_path_for(model_path: str) -> Path:
    """Return the _vecnorm.pkl path that CheckpointCallback writes.

    CheckpointCallback appends '_vecnorm.pkl' to the model stem, e.g.:
        runs/run_s1_XYZ/checkpoints/step_100000_vecnorm.pkl
    Path.with_suffix("") / "_vecnorm.pkl" would incorrectly create a
    subdirectory, so we build the path with string concatenation instead.
    """
    stem = str(Path(model_path).with_suffix(""))
    return Path(stem + "_vecnorm.pkl")


def _resolve_resume(model_path: str) -> tuple[str, Path]:
    """Return (sb3_load_path, vecnorm_path) for a resume checkpoint.

    eval_best is a directory containing best_model.zip written by EvalCallback;
    vecnorm is saved as a sibling file at {dir}_vecnorm.pkl by EvalWithVecNorm.
    All other checkpoints (checkpoints/step_N, final) are bare zip paths.
    """
    p = Path(model_path)
    if p.is_dir():
        return str(p / "best_model"), Path(str(p) + "_vecnorm.pkl")
    return model_path, _vecnorm_path_for(model_path)


load_dotenv()

# Categorical.__init__ validates every action's support on construction by default —
# measured at ~11% of a profiled training run (9,600 distribution constructions/rollout).
# MaskablePPO already constrains via action_masks; skip the redundant check.
torch.distributions.Distribution.set_default_validate_args(False)

# force_terminal keeps Rich colour/styling when PyCharm runs this as a Python
# config (stdout is not a TTY there, so auto-detection would strip ANSI).
# NO_COLOR still wins for plain piping / redirection.
console = Console(force_terminal=None if os.environ.get("NO_COLOR") else True)

_RL_ROOT = Path(__file__).resolve().parent.parent.parent  # rl/

os.environ.setdefault("ATB_RL_ROOT", str(_RL_ROOT))


register_configs()


def _env_option_gamma(vec_env) -> float | None:
    """Read the env's intra-option discount (reward.option_gamma) from the Rust core.

    Walks down VecEnv wrappers (e.g. VecNormalize) to the BatchVecEnv, whose
    ``.batch.reward_weights()`` returns (tick, pickup, deposit, wall_hit, option_gamma).
    Returns None if no introspectable batch env is found (check is then skipped).
    """
    env = vec_env
    for _ in range(8):
        if env is None:
            break
        if hasattr(env, "batch"):
            return float(env.batch.reward_weights()[4])
        env = getattr(env, "venv", None)
    return None


def _assert_gamma_consistency(vec_env, ppo_gamma: float) -> None:
    """Fail fast if PPO γ (yaml) and the env's intra-option γ (RON) disagree.

    These are the SAME per-tick discount applied at two levels of the option
    hierarchy: ``reward.option_gamma`` discounts ticks WITHIN an option (the Rust
    reward sum), and ``ppo.gamma`` discounts BETWEEN options (training/smdp.py
    raises it to γ^option_ticks). The semi-MDP formulation is only consistent when
    they are equal — but they live in different files (configs/ppo/*.yaml vs
    assets/world/config_stage*.ron), so a silent desync is easy. Treat option_gamma
    (the env) as the single source of truth and assert ppo.gamma matches it.
    """
    opt_gamma = _env_option_gamma(vec_env)
    if opt_gamma is None:
        return
    if not math.isclose(opt_gamma, ppo_gamma, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(
            f"gamma mismatch: ppo.gamma={ppo_gamma} (configs/ppo/*.yaml) != "
            f"reward.option_gamma={opt_gamma} (assets/world/config_stage*.ron). "
            "These must match — they are one per-tick discount at two levels of the "
            "option hierarchy (see training/smdp.py). Set ppo.gamma to the RON's "
            "option_gamma (or vice versa) so they agree."
        )


@hydra.main(config_path=str(_RL_ROOT / "configs"), config_name="train", version_base="1.3")
def train(cfg: DictConfig) -> None:
    raw: dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)  # type: ignore[assignment]
    t_cfg = TrainConfig(**raw["train"])
    p_cfg = PpoConfig(**raw["ppo"])
    e_cfg = EnvConfig(**raw["env"])

    algo_spec = get_algo(t_cfg.algo)
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    run_tag = run_dir.name
    _dev = get_device(t_cfg.device)
    if _dev.type == "cuda" and _dev.index is None:
        import torch
        device = f"cuda:{torch.cuda.current_device()}"
        torch.backends.cudnn.benchmark = True
    else:
        device = str(_dev)

    console.rule(f"[bold green]ATB Training — {run_tag}")
    console.print(Columns([
        kv_table("run", [
            ("algo", t_cfg.algo),
            ("stage", str(e_cfg.stage)),
            ("timesteps", f"{t_cfg.total_timesteps:,}"),
            ("n_envs", str(e_cfg.n_envs)),
            ("device", device),
            ("seed", str(e_cfg.seed)),
        ]),
        kv_table("ppo", [
            ("n_steps", str(p_cfg.n_steps)),
            ("batch", str(p_cfg.batch_size)),
            ("lr", f"{p_cfg.learning_rate:.0e} → {p_cfg.learning_rate_final:.0e}"),
            ("gamma", str(p_cfg.gamma)),
            ("ent_coef", f"{p_cfg.ent_coef} → {p_cfg.ent_coef_final}"),
            ("clip", str(p_cfg.clip_range)),
            ("target_kl", str(p_cfg.target_kl)),
        ]),
        kv_table("env", [
            ("config", Path(e_cfg.config_path).name),
            ("norm_reward", "✓" if e_cfg.normalize_reward else "✗"),
            ("max_steps", str(e_cfg.max_episode_steps or "—")),
        ]),
    ], equal=False, expand=False))

    # Environments.
    vec_env = build_vec_env(e_cfg, gamma=p_cfg.gamma)
    # One per-tick discount, two config files — fail fast if they desync.
    _assert_gamma_consistency(vec_env, p_cfg.gamma)
    vec_normalize = vec_env if isinstance(vec_env, VecNormalize) else None
    eval_env = build_vec_env(e_cfg, eval_mode=True, gamma=p_cfg.gamma)
    eval_normalize = eval_env if isinstance(eval_env, VecNormalize) else None

    # Share live running-stat references so the eval env always uses the same
    # obs_rms/ret_rms objects as training. Without this, eval_env starts with
    # freshly-initialised stats (mean=0, var=1) causing the agent to act
    # randomly in eval — producing the flat -5.00 reward (pure tick penalty).
    _sync_vecnorm_stats(vec_normalize, eval_normalize)

    # Schedules.
    lr_schedule = linear_schedule(p_cfg.learning_rate, p_cfg.learning_rate_final)
    clip_schedule = linear_schedule(p_cfg.clip_range, p_cfg.clip_range_final)
    ent_schedule = linear_schedule(p_cfg.ent_coef, p_cfg.ent_coef_final)

    resume: str | None = getattr(t_cfg, "resume", None)

    if resume:
        resume_path, vn_path = _resolve_resume(resume)
        console.print(f"Resuming from {resume}")
        model = algo_spec.cls.load(
            resume_path,
            env=vec_env,
            device=t_cfg.device,
            tensorboard_log=str(run_dir / "tensorboard"),
        )
        # load() restores the checkpoint's hyperparameters/schedules — the yaml
        # values in the table above do NOT apply. Report what actually runs.
        hp = {k: getattr(model, k, None)
              for k in ("n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda", "target_kl")}
        console.print(
            "  [yellow]checkpoint overrides yaml ppo table:[/yellow] "
            + " ".join(f"{k}={v}" for k, v in hp.items())
        )
        if vec_normalize is not None and vn_path.exists():
            # Copy saved running stats into the existing VecNormalize rather than
            # creating a new wrapper around it (which would double-normalize obs).
            # The model's env reference to vec_normalize remains valid.
            _loaded_vn = VecNormalize.load(str(vn_path), cast(VecEnv, vec_normalize.venv))
            if _loaded_vn.norm_obs:
                vec_normalize.obs_rms = _loaded_vn.obs_rms
            vec_normalize.ret_rms = _loaded_vn.ret_rms
            console.print(f"  vecnorm  ← {vn_path}")
        elif vec_normalize is not None:
            console.print(f"  [yellow]vecnorm not found at {vn_path}, using fresh stats[/yellow]")
        # Re-sync eval stats after potentially reloading vec_normalize.
        eval_normalize = eval_env if isinstance(eval_env, VecNormalize) else None
        _sync_vecnorm_stats(vec_normalize, eval_normalize)
    else:
        policy_kwargs = {
            **ATB_POLICY_KWARGS,
            "net_arch": {"pi": list(p_cfg.net_arch_pi), "vf": list(p_cfg.net_arch_vf)},
        }
        model = algo_spec.constructor(
            policy=algo_spec.policy,
            env=vec_env,
            # Semi-MDP γ^k cross-option discounting (None → plain γ for unsupported algos).
            rollout_buffer_class=smdp_buffer_class(t_cfg.algo),
            learning_rate=lr_schedule,
            n_steps=p_cfg.n_steps,
            batch_size=p_cfg.batch_size,
            n_epochs=p_cfg.n_epochs,
            gamma=p_cfg.gamma,
            gae_lambda=p_cfg.gae_lambda,
            clip_range=clip_schedule,
            ent_coef=p_cfg.ent_coef,
            vf_coef=p_cfg.vf_coef,
            max_grad_norm=p_cfg.max_grad_norm,
            target_kl=p_cfg.target_kl,
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(run_dir / "tensorboard"),
            verbose=0,
            seed=e_cfg.seed,
            device=t_cfg.device,
        )

    n_params = sum(p.numel() for p in model.policy.parameters())
    console.print(f"  parameters      : {n_params:,}")
    _print_arch(model)

    eval_best_path = str(run_dir / "eval_best")

    callbacks = [
        CheckpointCallback(
            save_freq=t_cfg.checkpoint_freq,
            ckpt_dir=ckpt_dir,
            vec_normalize=vec_normalize,
            # ONNX export happens on eval-best only (make_eval_callback below):
            # a synchronous deepcopy + onnxruntime validation per periodic
            # checkpoint stalls the GPU-update-bound training loop for seconds.
            onnx_export=False,
        ),
        stats_callback := EpisodeStatsCallback(stats_path=run_dir / "stats.h5"),
        # Logs chosen-gold-region distance + own-region skip rate to TB (policy/*).
        PolicyTelemetryCallback(),
        # Feeds per-step option lengths to the SMDP rollout buffer (γ^k discount).
        SmdpDiscountCallback(),
        RichLogCallback(console),
        EntropyCoefScheduleCallback(schedule=ent_schedule),
        # make_eval_callback saves vec_normalize alongside every eval-best model
        # (SB3's stock EvalCallback only saves the zip, leaving no matching
        # _vecnorm.pkl) and selects the maskable vs standard eval base by algo so
        # masking-only kwargs are never passed to a non-maskable model's predict().
        make_eval_callback(
            t_cfg.algo,
            eval_env=eval_env,
            best_model_save_path=eval_best_path,
            log_path=str(run_dir / "eval_log"),
            eval_freq=max(t_cfg.eval_freq // max(e_cfg.n_envs, 1), 1),
            # Episode count comes from cfg (default 10) — 3 was too noisy
            # given high episode variance (std ~9 in the first run).
            n_eval_episodes=t_cfg.eval_episodes,
            deterministic=True,
            render=False,
            verbose=0,
            vec_normalize=vec_normalize,
            onnx_path=run_dir / "models" / "eval_best" / "policy.onnx",
        ),
    ]

    # Periodic in-training plot refresh (non-blocking). 0 = only at stage end.
    if getattr(t_cfg, "plot_freq", 0) and t_cfg.plot_freq > 0:
        callbacks.append(PeriodicPlotCallback(
            plot_freq=t_cfg.plot_freq,
            run_dir=run_dir,
            plot_script=_RL_ROOT / "scripts" / "plot_runs.py",
        ))
        console.print(f"  plots           : every {t_cfg.plot_freq:,} steps → {run_dir.name}/plots/")

    t0 = time.time()
    interrupted = False
    try:
        model.learn(
            total_timesteps=t_cfg.total_timesteps,
            callback=callbacks,
            tb_log_name=run_tag,
            reset_num_timesteps=resume is None,
        )
    except KeyboardInterrupt:
        interrupted = True
        console.print("[yellow]Interrupted — saving checkpoint and exporting ONNX...[/yellow]")
    finally:
        stats_callback.close()
        vec_env.close()
        eval_env.close()

    elapsed = time.time() - t0
    if interrupted:
        console.print(f"[bold yellow]Stopped after {elapsed / 3600:.2f}h[/bold yellow]")
    else:
        console.print(f"[bold green]Done in {elapsed / 3600:.2f}h[/bold green]")

    final = run_dir / "final"
    model.save(str(final))
    if vec_normalize is not None:
        vec_normalize.save(str(final) + "_vecnorm.pkl")

    console.print(f"  model  → {final}.zip")
    console.print(f"  stats  → {run_dir / 'stats.h5'}")
    console.print(f"  tb     → tensorboard --logdir {run_dir / 'tensorboard'}")

    # Export the best available policy to models/eval_best/policy.onnx.
    # The eval callback already writes this on every new best during training;
    # this is a fallback for when training ended before the first eval ran.
    onnx_path = run_dir / "models" / "eval_best" / "policy.onnx"
    if not onnx_path.exists():
        eval_best_zip = Path(eval_best_path) / "best_model.zip"
        if eval_best_zip.exists():
            export_src = Path(eval_best_path) / "best_model"
            export_vn = Path(eval_best_path + "_vecnorm.pkl")
            export_label = "eval best"
        else:
            export_src = final
            export_vn = Path(str(final) + "_vecnorm.pkl")
            export_label = "final (no eval-best yet)"
        try:
            best_policy = algo_spec.cls.load(str(export_src)).policy
            onnx_path.parent.mkdir(parents=True, exist_ok=True)
            export_to_onnx(best_policy, onnx_path,
                           vecnorm_path=export_vn if export_vn.exists() else None)
            console.print(f"  policy → {onnx_path}  ({export_label})")
        except Exception as exc:
            console.print(f"  [yellow]ONNX export failed: {exc}[/yellow]")
    else:
        console.print(f"  policy → {onnx_path}  (written by eval callback)")


if __name__ == "__main__":
    train()  # type: ignore[call-arg]  # Hydra injects cfg at runtime
