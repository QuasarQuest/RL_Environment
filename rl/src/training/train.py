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
    atb-train +train.resume=runs/models/run_s1_XYZ_step_100000
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.table import Table
import torch.nn as nn
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecNormalize

from env.factory import build_vec_env
from model.policy import ATB_POLICY_KWARGS
from training.algos import get_algo
from training.callbacks import (
    CheckpointCallback,
    EntropyCoefScheduleCallback,
    EpisodeStatsCallback,
    RichLogCallback,
)
from training.config import EnvConfig, PpoConfig, TrainConfig, register_configs


def _kv_table(title: str, rows: list[tuple[str, str]]) -> Table:
    t = Table(title=title, box=box.SIMPLE, show_header=False,
              title_style="bold cyan", padding=(0, 1))
    t.add_column(style="dim", no_wrap=True)
    t.add_column(justify="right", no_wrap=True)
    for k, v in rows:
        t.add_row(k, v)
    return t


def linear_schedule(start: float, end: float):
    if start == end:
        return lambda _: float(start)
    return lambda p: float(end + p * (start - end))


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
        # activations omitted — they're noise at this level
    return " → ".join(parts)


def _print_arch(model) -> None:
    policy = model.policy
    fe = policy.features_extractor
    obs = tuple(policy.observation_space.shape)

    if hasattr(fe, "conv") and hasattr(fe, "head"):
        extractor = f"CNN  obs={obs}  {_seq_str(fe.conv)} → {_seq_str(fe.head)}"
    elif hasattr(fe, "net"):
        extractor = f"MLP  obs={obs}  {_seq_str(fe.net)}"
    else:
        extractor = f"{type(fe).__name__}  obs={obs}"

    mlp = policy.mlp_extractor
    pi_mid = _seq_str(mlp.policy_net)
    vf_mid = _seq_str(mlp.value_net)
    pi_str = (f"{pi_mid} → " if pi_mid else "") + f"Linear(→{policy.action_net.out_features})"
    vf_str = (f"{vf_mid} → " if vf_mid else "") + f"Linear(→1)"

    console.print(f"  [dim]extractor[/dim]       : {extractor}")
    console.print(f"  [dim]pi[/dim]              : {pi_str}")
    console.print(f"  [dim]vf[/dim]              : {vf_str}")


# Load secrets from .env before anything else.
# WANDB_API_KEY, ATB_DEVICE, etc. are available via os.environ after this.
load_dotenv()

console = Console()

_RL_ROOT = Path(__file__).resolve().parent.parent.parent  # rl/

# Expose rl/ as an env var so train.yaml can reference it in hydra.run.dir.
# Must be set before Hydra reads the YAML, which happens after module-level
# code runs — so this placement is correct.
os.environ.setdefault("ATB_RL_ROOT", str(_RL_ROOT))


def _resolve_dirs(cfg: TrainConfig) -> tuple[Path, Path, Path]:
    base = _RL_ROOT
    models = base / cfg.models_dir
    stats = base / cfg.stats_dir
    tb = base / cfg.tensorboard_dir
    for d in (models, stats, tb):
        d.mkdir(parents=True, exist_ok=True)
    return models, stats, tb


# Register structured configs before Hydra initialises.
register_configs()


@hydra.main(config_path=str(_RL_ROOT / "configs"), config_name="train", version_base="1.3")
def train(cfg: DictConfig) -> None:
    # OmegaConf.to_container gives a plain dict; construct typed dataclasses
    # explicitly so downstream code gets attribute access + the config_path property.
    raw: dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)  # type: ignore[assignment]
    t_cfg = TrainConfig(**raw["train"])
    p_cfg = PpoConfig(**raw["ppo"])
    e_cfg = EnvConfig(**raw["env"])

    algo_spec = get_algo(t_cfg.algo)
    run_tag = f"{t_cfg.run_name}_s{e_cfg.stage}_{int(time.time())}"
    models_dir, stats_dir, tb_dir = _resolve_dirs(t_cfg)
    _dev = get_device(t_cfg.device)
    if _dev.type == "cuda" and _dev.index is None:
        import torch
        device = f"cuda:{torch.cuda.current_device()}"
    else:
        device = str(_dev)

    console.rule(f"[bold green]ATB Training — {run_tag}")
    console.print(Columns([
        _kv_table("run", [
            ("algo",      t_cfg.algo),
            ("stage",     str(e_cfg.stage)),
            ("timesteps", f"{t_cfg.total_timesteps:,}"),
            ("n_envs",    str(e_cfg.n_envs)),
            ("device",    device),
            ("seed",      str(e_cfg.seed)),
        ]),
        _kv_table("ppo", [
            ("n_steps",   str(p_cfg.n_steps)),
            ("batch",     str(p_cfg.batch_size)),
            ("lr",        f"{p_cfg.learning_rate:.0e} → {p_cfg.learning_rate_final:.0e}"),
            ("gamma",     str(p_cfg.gamma)),
            ("ent_coef",  f"{p_cfg.ent_coef} → {p_cfg.ent_coef_final}"),
            ("clip",      str(p_cfg.clip_range)),
            ("target_kl", str(p_cfg.target_kl)),
        ]),
        _kv_table("env", [
            ("norm_reward", "✓" if e_cfg.normalize_reward else "✗"),
            ("clip_reward", "✓" if e_cfg.clip_reward else "✗"),
            ("clip_max",    str(e_cfg.clip_reward_max)),
            ("max_steps",   str(e_cfg.max_episode_steps or "—")),
        ]),
    ], equal=False, expand=False))

    # Environments.
    vec_env = build_vec_env(e_cfg)
    vec_normalize = vec_env if isinstance(vec_env, VecNormalize) else None
    eval_env = build_vec_env(e_cfg, eval_mode=True)

    # Schedules.
    lr_schedule = linear_schedule(p_cfg.learning_rate, p_cfg.learning_rate_final)
    clip_schedule = linear_schedule(p_cfg.clip_range, p_cfg.clip_range_final)
    ent_schedule = linear_schedule(p_cfg.ent_coef, p_cfg.ent_coef_final)

    resume: str | None = getattr(t_cfg, "resume", None)

    if resume:
        console.print(f"Resuming from {resume}")
        model = algo_spec.cls.load(
            resume,
            env=vec_env,
            device=t_cfg.device,
            tensorboard_log=str(tb_dir),
        )
    else:
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
            policy_kwargs=ATB_POLICY_KWARGS,
            tensorboard_log=str(tb_dir),
            verbose=0,
            seed=e_cfg.seed,
            device=t_cfg.device,
        )

    n_params = sum(p.numel() for p in model.policy.parameters())
    console.print(f"  parameters      : {n_params:,}")
    _print_arch(model)

    callbacks = [
        CheckpointCallback(
            save_freq=t_cfg.checkpoint_freq,
            models_dir=models_dir,
            run_name=run_tag,
            vec_normalize=vec_normalize,
        ),
        EpisodeStatsCallback(stats_path=stats_dir / f"{run_tag}.h5"),
        RichLogCallback(console),
        EntropyCoefScheduleCallback(
            schedule=ent_schedule,
            total_timesteps=t_cfg.total_timesteps,
        ),
        EvalCallback(
            eval_env=eval_env,
            best_model_save_path=str(models_dir / f"{run_tag}_eval_best"),
            log_path=str(stats_dir / f"{run_tag}_eval"),
            eval_freq=max(t_cfg.eval_freq // max(e_cfg.n_envs, 1), 1),
            n_eval_episodes=t_cfg.eval_episodes,
            deterministic=True,
            render=False,
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

    final = models_dir / f"{run_tag}_final"
    model.save(str(final))
    if vec_normalize is not None:
        vec_normalize.save(str(final) + "_vecnorm.pkl")

    console.print(f"  model  → {final}.zip")
    console.print(f"  stats  → {stats_dir / run_tag}.h5")
    console.print(f"  tb     → tensorboard --logdir {tb_dir}")


if __name__ == "__main__":
    train()  # type: ignore[call-arg]  # Hydra injects cfg at runtime

