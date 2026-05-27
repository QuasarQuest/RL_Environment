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
import shutil
import time
from pathlib import Path

import hydra
import torch.nn as nn
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.table import Table
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecNormalize

from env.factory import build_vec_env
from network.export import export_to_onnx
from network.policy import ATB_POLICY_KWARGS, ATB_RECURRENT_POLICY_KWARGS
from training.algos import get_algo
from training.callbacks import (
    CheckpointCallback,
    EntropyCoefScheduleCallback,
    EpisodeStatsCallback,
    EvalWithVecNorm,
    RichLogCallback,
    kv_table,
)
from training.config import EnvConfig, PpoConfig, TrainConfig, register_configs
from training.schedules import linear_schedule


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

    if hasattr(policy, "lstm_actor"):
        lstm = policy.lstm_actor
        console.print(f"  [dim]lstm[/dim]             : hidden={lstm.hidden_size}  layers={lstm.num_layers}")
        if hasattr(policy, "action_net"):
            console.print(f"  [dim]pi[/dim]              : Linear(→{policy.action_net.out_features})")
        console.print(f"  [dim]vf[/dim]              : Linear(→1)")
    elif hasattr(policy, "mlp_extractor"):
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

console = Console()

_RL_ROOT = Path(__file__).resolve().parent.parent.parent  # rl/
_WORKSPACE_ROOT = _RL_ROOT.parent                         # algorithm_test_bed/

os.environ.setdefault("ATB_RL_ROOT", str(_RL_ROOT))


def _resolve_dirs(cfg: TrainConfig, run_tag: str) -> tuple[Path, Path]:
    """Create and return (run_dir, ckpt_dir) for this run.

    Layout inside run_dir:
        checkpoints/   — step_N and best_rolling snapshots
        eval_best/     — best model from EvalCallback (created by SB3)
        tensorboard/   — TensorBoard event files (created by SB3)
        stats.h5       — per-episode HDF5 stats
        eval_log/      — EvalCallback's evaluations.npz (created by SB3)
        final.zip      — model at end of training
        policy.onnx    — ONNX export of eval-best policy
    """
    run_dir = _WORKSPACE_ROOT / cfg.output_dir / run_tag
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, ckpt_dir


register_configs()


@hydra.main(config_path=str(_RL_ROOT / "configs"), config_name="train", version_base="1.3")
def train(cfg: DictConfig) -> None:
    raw: dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)  # type: ignore[assignment]
    t_cfg = TrainConfig(**raw["train"])
    p_cfg = PpoConfig(**raw["ppo"])
    e_cfg = EnvConfig(**raw["env"])

    algo_spec = get_algo(t_cfg.algo)
    run_tag = f"{t_cfg.run_name}_s{e_cfg.stage}_{int(time.time())}"
    run_dir, ckpt_dir = _resolve_dirs(t_cfg, run_tag)
    _dev = get_device(t_cfg.device)
    if _dev.type == "cuda" and _dev.index is None:
        import torch
        device = f"cuda:{torch.cuda.current_device()}"
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
            ("clip_reward", "✓" if e_cfg.clip_reward else "✗"),
            ("clip_max", str(e_cfg.clip_reward_max)),
            ("max_steps", str(e_cfg.max_episode_steps or "—")),
        ]),
    ], equal=False, expand=False))

    # Environments.
    vec_env = build_vec_env(e_cfg)
    vec_normalize = vec_env if isinstance(vec_env, VecNormalize) else None
    eval_env = build_vec_env(e_cfg, eval_mode=True)
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
        if vec_normalize is not None and vn_path.exists():
            vec_normalize = VecNormalize.load(str(vn_path), vec_env)
            console.print(f"  vecnorm  ← {vn_path}")
        else:
            if vec_normalize is not None:
                console.print(f"  [yellow]vecnorm not found at {vn_path}, using fresh stats[/yellow]")
        # Re-sync eval stats after potentially reloading vec_normalize.
        eval_normalize = eval_env if isinstance(eval_env, VecNormalize) else None
        _sync_vecnorm_stats(vec_normalize, eval_normalize)
    else:
        if t_cfg.algo == "recurrent_ppo":
            policy_kwargs = {
                **ATB_RECURRENT_POLICY_KWARGS,
                "lstm_hidden_size": p_cfg.lstm_hidden_size,
                "n_lstm_layers": p_cfg.n_lstm_layers,
            }
        else:
            policy_kwargs = {
                **ATB_POLICY_KWARGS,
                "net_arch": dict(pi=list(p_cfg.net_arch_pi), vf=list(p_cfg.net_arch_vf)),
            }
        model = algo_spec.constructor(
            policy=algo_spec.policy,
            env=vec_env,
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
        ),
        EpisodeStatsCallback(stats_path=run_dir / "stats.h5"),
        RichLogCallback(console),
        EntropyCoefScheduleCallback(
            schedule=ent_schedule,
            total_timesteps=t_cfg.total_timesteps,
        ),
        # FIX: EvalWithVecNorm saves vec_normalize alongside every eval-best
        # model. SB3's stock EvalCallback only saves the model zip, leaving no
        # matching _vecnorm.pkl — the best model cannot be loaded correctly
        # for deployment or resume without the normalisation stats.
        EvalWithVecNorm(
            eval_env=eval_env,
            best_model_save_path=eval_best_path,
            log_path=str(run_dir / "eval_log"),
            eval_freq=max(t_cfg.eval_freq // max(e_cfg.n_envs, 1), 1),
            # FIX: increased from 3 to 20 — with high episode variance
            # (std ~9 in the previous run) 3 episodes give unreliable signal.
            n_eval_episodes=t_cfg.eval_episodes,
            deterministic=True,
            render=False,
            vec_normalize=vec_normalize,
        ),
    ]

    t0 = time.time()
    try:
        model.learn(
            total_timesteps=t_cfg.total_timesteps,
            callback=callbacks,
            tb_log_name=run_tag,
            reset_num_timesteps=resume is None,
        )
    finally:
        vec_env.close()
        eval_env.close()

    elapsed = time.time() - t0
    console.print(f"[bold green]Done in {elapsed / 3600:.2f}h[/bold green]")

    final = run_dir / "final"
    model.save(str(final))
    if vec_normalize is not None:
        vec_normalize.save(str(final) + "_vecnorm.pkl")

    console.print(f"  model  → {final}.zip")
    console.print(f"  stats  → {run_dir / 'stats.h5'}")
    console.print(f"  tb     → tensorboard --logdir {run_dir / 'tensorboard'}")

    onnx_path = run_dir / "policy.onnx"
    best_vn_path = Path(eval_best_path + "_vecnorm.pkl")
    try:
        best_policy = algo_spec.cls.load(str(Path(eval_best_path) / "best_model")).policy
        export_to_onnx(best_policy, onnx_path, vecnorm_path=best_vn_path if best_vn_path.exists() else None)
        console.print(f"  policy → {onnx_path}  (eval best)")
        viewer_path = _WORKSPACE_ROOT / "assets" / "model" / "policy.onnx"
        shutil.copy2(onnx_path, viewer_path)
        console.print(f"  viewer → {viewer_path}  (copied)")
    except Exception as exc:
        console.print(f"  [yellow]ONNX export failed: {exc}[/yellow]")


if __name__ == "__main__":
    train()  # type: ignore[call-arg]  # Hydra injects cfg at runtime
