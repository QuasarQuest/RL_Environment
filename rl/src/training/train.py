"""Training entry point. Algorithm-agnostic via the registry in `algos.py`."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecNormalize

from env.factory import build_vec_env
from model.policy import ATB_POLICY_KWARGS
from training.algos import get_algo
from training.callbacks import (
    CheckpointCallback,
    EntropyCoefScheduleCallback,
    EpisodeStatsCallback,
)
from training.config import TrainConfig
from training.schedules import linear_schedule

console = Console()
app = typer.Typer()

_RL_ROOT = Path(__file__).resolve().parent.parent


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
    algo: str = typer.Option("ppo", "--algo"),
    n_envs: int = typer.Option(1, "--n-envs"),
) -> None:
    cfg = TrainConfig(
        stage=stage,
        total_timesteps=total_timesteps,
        run_name=run_name,
        device=device,
        seed=seed,
        algo=algo,
        n_envs=n_envs,
    )
    ppo = cfg.ppo
    algo_spec = get_algo(cfg.algo)
    run_tag = f"{run_name}_s{stage}_{int(time.time())}"
    models_dir, stats_dir, tb_dir = _resolve_dirs(cfg)

    console.rule(f"[bold green]ATB Training — {run_tag}")
    console.print(f"  algo            : {cfg.algo}")
    console.print(f"  stage           : {stage}")
    console.print(f"  config          : {cfg.config_path}")
    console.print(f"  total_timesteps : {total_timesteps:,}")
    console.print(f"  n_envs          : {n_envs}")
    console.print(f"  device          : {device}")
    console.print(f"  seed            : {seed}")

    # Training vec env (may use VecNormalize in training mode).
    vec_env = build_vec_env(cfg)
    vec_normalize = vec_env if isinstance(vec_env, VecNormalize) else None

    # Separate eval env. We deep-copy the normaliser's running stats so eval
    # sees the same input distribution but does NOT update them.
    eval_env = build_vec_env(cfg, eval_mode=True)
    if isinstance(eval_env, VecNormalize) and vec_normalize is not None:
        eval_env.obs_rms = vec_normalize.obs_rms
        eval_env.ret_rms = vec_normalize.ret_rms

    # Schedules: PPO consumes lr and clip_range as callables; ent_coef needs
    # the dedicated callback because PPO reads it as a plain float.
    lr_schedule = linear_schedule(ppo.learning_rate, ppo.learning_rate_final)
    clip_schedule = linear_schedule(ppo.clip_range, ppo.clip_range_final)
    ent_schedule = linear_schedule(ppo.ent_coef, ppo.ent_coef_final)

    if resume:
        console.print(f"Resuming from {resume}")
        model = algo_spec.cls.load(
            resume,
            env=vec_env,
            device=device,
            tensorboard_log=str(tb_dir),
        )
    else:
        model = algo_spec.cls(
            policy=algo_spec.policy,
            env=vec_env,
            learning_rate=lr_schedule,
            n_steps=ppo.n_steps,
            batch_size=ppo.batch_size,
            n_epochs=ppo.n_epochs,
            gamma=ppo.gamma,
            gae_lambda=ppo.gae_lambda,
            clip_range=clip_schedule,
            ent_coef=ppo.ent_coef,  # initial value; callback updates it
            vf_coef=ppo.vf_coef,
            max_grad_norm=ppo.max_grad_norm,
            policy_kwargs=ATB_POLICY_KWARGS,
            tensorboard_log=str(tb_dir),
            verbose=1,
            seed=seed,
            device=device,
        )

    n_params = sum(p.numel() for p in model.policy.parameters())
    console.print(f"  parameters      : {n_params:,}")

    callbacks = [
        CheckpointCallback(
            save_freq=cfg.checkpoint_freq,
            models_dir=models_dir,
            run_name=run_tag,
            vec_normalize=vec_normalize,
        ),
        EpisodeStatsCallback(stats_path=stats_dir / f"{run_tag}.h5"),
        EntropyCoefScheduleCallback(
            schedule=ent_schedule,
            total_timesteps=total_timesteps,
        ),
        EvalCallback(
            eval_env=eval_env,
            best_model_save_path=str(models_dir / f"{run_tag}_eval_best"),
            log_path=str(stats_dir / f"{run_tag}_eval"),
            eval_freq=max(cfg.eval_freq // max(n_envs, 1), 1),
            n_eval_episodes=cfg.eval_episodes,
            deterministic=True,
            render=False,
        ),
    ]

    t0 = time.time()
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            tb_log_name=run_tag,
            reset_num_timesteps=resume is None,
        )
    finally:
        # Ensure envs are torn down even on Ctrl-C, otherwise SubprocVecEnv
        # leaves worker processes around.
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
    app()
