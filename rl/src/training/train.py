# rl/src/training/train.py

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

# ── Ensure project root is on sys.path regardless of cwd ─────────────────────
# Resolves to rl/ directory so `from env.atb_env import ...` works.
_HERE = Path(__file__).resolve().parent          # rl/src/training/
_SRC  = _HERE.parent                             # rl/src/
_RL   = _SRC.parent                             # rl/
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_RL) not in sys.path:
    sys.path.insert(0, str(_RL))
# ─────────────────────────────────────────────────────────────────────────────

import typer
from rich.console import Console
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from env.atb_env import AtbEnv
from env.wrappers import ClipRewardWrapper, RewardScaleWrapper
from model.policy import ATB_POLICY_KWARGS
from training.config import TrainConfig, PpoConfig
from training.callbacks import CheckpointCallback, EpisodeStatsCallback

console = Console()
app     = typer.Typer()


def make_env(cfg: TrainConfig):
    env = AtbEnv()
    if cfg.clip_reward:
        env = ClipRewardWrapper(env, max_abs=cfg.clip_reward_max)
    if cfg.reward_scale != 1.0:
        env = RewardScaleWrapper(env, scale=cfg.reward_scale)
    env = Monitor(env)
    return env


def build_vec_env(cfg: TrainConfig):
    vec_env = DummyVecEnv([lambda: make_env(cfg)])
    if cfg.normalize_obs:
        vec_env = VecNormalize(
            vec_env,
            norm_obs    = cfg.normalize_obs,
            norm_reward = cfg.normalize_reward,
            clip_obs    = 10.0,
        )
    return vec_env


@app.command()
def train(
    total_timesteps: int           = typer.Option(10_000_000, "--total-timesteps"),
    run_name:        str           = typer.Option("run",       "--run-name"),
    resume:          Optional[str] = typer.Option(None,        "--resume"),
    device:          str           = typer.Option("cpu",      "--device"),
    seed:            int           = typer.Option(42,          "--seed"),
):
    """Train a PPO agent on the Algorithm Test Bed simulation."""

    cfg     = TrainConfig(
        total_timesteps = total_timesteps,
        run_name        = run_name,
        device          = device,
        seed            = seed,
    )
    ppo_cfg    = cfg.ppo
    run_tag    = f"{run_name}_{int(time.time())}"
    models_dir = _RL / cfg.models_dir
    stats_dir  = _RL / cfg.stats_dir
    tb_dir     = _RL / cfg.tensorboard_dir

    for d in [models_dir, stats_dir, tb_dir]:
        d.mkdir(parents=True, exist_ok=True)

    console.rule(f"[bold green]ATB Training — {run_tag}")
    console.print(f"  total_timesteps : {total_timesteps:,}")
    console.print(f"  device          : {device}")
    console.print(f"  seed            : {seed}")

    console.print("\nBuilding environment ...")
    vec_env = build_vec_env(cfg)

    if resume:
        console.print(f"Resuming from {resume} ...")
        model = PPO.load(
            resume,
            env             = vec_env,
            device          = device,
            tensorboard_log = str(tb_dir),
        )
    else:
        model = PPO(
            policy          = "MlpPolicy",
            env             = vec_env,
            learning_rate   = ppo_cfg.learning_rate,
            n_steps         = ppo_cfg.n_steps,
            batch_size      = ppo_cfg.batch_size,
            n_epochs        = ppo_cfg.n_epochs,
            gamma           = ppo_cfg.gamma,
            gae_lambda      = ppo_cfg.gae_lambda,
            clip_range      = ppo_cfg.clip_range,
            ent_coef        = ppo_cfg.ent_coef,
            vf_coef         = ppo_cfg.vf_coef,
            max_grad_norm   = ppo_cfg.max_grad_norm,
            policy_kwargs   = ATB_POLICY_KWARGS,
            tensorboard_log = str(tb_dir),
            verbose         = 1,
            seed            = seed,
            device          = device,
        )

    console.print(f"  Parameters: {sum(p.numel() for p in model.policy.parameters()):,}")

    vec_normalize = vec_env if isinstance(vec_env, VecNormalize) else None
    checkpoint_cb = CheckpointCallback(
        save_freq     = cfg.checkpoint_freq,
        models_dir    = str(models_dir),
        run_name      = run_tag,
        vec_normalize = vec_normalize,
        verbose       = 1,
    )
    stats_cb = EpisodeStatsCallback(
        stats_dir = str(stats_dir),
        run_name  = run_tag,
    )

    console.print("\n[bold]Starting training ...[/bold]")
    t0 = time.time()

    model.learn(
        total_timesteps     = total_timesteps,
        callback            = [checkpoint_cb, stats_cb],
        tb_log_name         = run_tag,
        reset_num_timesteps = resume is None,
    )

    elapsed = time.time() - t0
    console.print(f"\n[bold green]Training complete in {elapsed/3600:.2f}h[/bold green]")

    final_path = models_dir / f"{run_tag}_final"
    model.save(str(final_path))
    if vec_normalize is not None:
        vec_normalize.save(str(final_path) + "_vecnorm.pkl")

    console.print(f"  Model   → {final_path}.zip")
    console.print(f"  Stats   → {stats_dir / run_tag}.h5")
    console.print(f"  TB logs → tensorboard --logdir {tb_dir}")


if __name__ == "__main__":
    app()