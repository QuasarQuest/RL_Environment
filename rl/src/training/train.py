from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional

import typer
from rich.console import Console
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env.atb_env import AtbEnv
from env.wrappers import ClipRewardWrapper, RewardScaleWrapper
from model.policy import ATB_POLICY_KWARGS
from training.callbacks import CheckpointCallback, EpisodeStatsCallback
from training.config import TrainConfig

console = Console()
app = typer.Typer()

_RL_ROOT = Path(__file__).resolve().parent.parent


def _linear_schedule(start: float, end: float) -> Callable[[float], float]:
    def schedule(progress: float) -> float:
        return end + progress * (start - end)
    return schedule


def _make_env(cfg: TrainConfig) -> gym.Env:
    from gymnasium import Env
    env: Env = AtbEnv(cfg.config_path)
    if cfg.clip_reward:
        env = ClipRewardWrapper(env, max_abs=cfg.clip_reward_max)
    if cfg.reward_scale != 1.0:
        env = RewardScaleWrapper(env, scale=cfg.reward_scale)
    return Monitor(env)


def _build_vec_env(cfg: TrainConfig) -> DummyVecEnv | VecNormalize:
    vec_env = DummyVecEnv([lambda: _make_env(cfg)])
    if cfg.normalize_obs or cfg.normalize_reward:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=cfg.normalize_obs,
            norm_reward=cfg.normalize_reward,
            clip_obs=10.0,
            clip_reward=10.0,
        )
    return vec_env


def _resolve_dirs(cfg: TrainConfig) -> tuple[Path, Path, Path]:
    base = _RL_ROOT
    models = base / cfg.models_dir
    stats = base / cfg.stats_dir
    tb = base / cfg.tensorboard_dir
    for d in (models, stats, tb):
        d.mkdir(parents=True, exist_ok=True)
    return models, stats, tb


@app.command()
def train(
    stage: int = typer.Option(1, "--stage"),
    total_timesteps: int = typer.Option(2_000_000, "--total-timesteps"),
    run_name: str = typer.Option("run", "--run-name"),
    resume: Optional[str] = typer.Option(None, "--resume"),
    device: str = typer.Option("cpu", "--device"),
    seed: int = typer.Option(42, "--seed"),
) -> None:
    cfg = TrainConfig(
        stage=stage,
        total_timesteps=total_timesteps,
        run_name=run_name,
        device=device,
        seed=seed,
    )
    ppo = cfg.ppo
    run_tag = f"{run_name}_s{stage}_{int(time.time())}"
    models_dir, stats_dir, tb_dir = _resolve_dirs(cfg)

    console.rule(f"[bold green]ATB Training — {run_tag}")
    console.print(f"  stage           : {stage}")
    console.print(f"  config          : {cfg.config_path}")
    console.print(f"  total_timesteps : {total_timesteps:,}")
    console.print(f"  device          : {device}")
    console.print(f"  seed            : {seed}")

    vec_env = _build_vec_env(cfg)

    lr_schedule = _linear_schedule(ppo.learning_rate, ppo.learning_rate_final)
    ent_schedule = _linear_schedule(ppo.ent_coef, ppo.ent_coef_final)

    if resume:
        console.print(f"Resuming from {resume}")
        model = PPO.load(resume, env=vec_env, device=device, tensorboard_log=str(tb_dir))
    else:
        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=lr_schedule,
            n_steps=ppo.n_steps,
            batch_size=ppo.batch_size,
            n_epochs=ppo.n_epochs,
            gamma=ppo.gamma,
            gae_lambda=ppo.gae_lambda,
            clip_range=ppo.clip_range,
            ent_coef=ent_schedule,
            vf_coef=ppo.vf_coef,
            max_grad_norm=ppo.max_grad_norm,
            policy_kwargs=ATB_POLICY_KWARGS,
            tensorboard_log=str(tb_dir),
            verbose=1,
            seed=seed,
            device=device,
        )

    console.print(f"  parameters      : {sum(p.numel() for p in model.policy.parameters()):,}")

    vec_normalize = vec_env if isinstance(vec_env, VecNormalize) else None
    callbacks = [
        CheckpointCallback(
            save_freq=cfg.checkpoint_freq,
            models_dir=models_dir,
            run_name=run_tag,
            vec_normalize=vec_normalize,
        ),
        EpisodeStatsCallback(stats_path=stats_dir / f"{run_tag}.h5"),
    ]

    t0 = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        tb_log_name=run_tag,
        reset_num_timesteps=resume is None,
    )

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
    app()