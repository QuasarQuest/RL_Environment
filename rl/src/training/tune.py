from __future__ import annotations

from pathlib import Path
from typing import Any

import optuna
import typer
from optuna.samplers import TPESampler
from rich.console import Console
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env.atb_env import AtbEnv
from env.wrappers import ClipRewardWrapper
from model.policy import ATB_POLICY_KWARGS
from training.config import TrainConfig, TuneConfig

console = Console()
app = typer.Typer()

_RL_ROOT = Path(__file__).resolve().parent.parent


def _sample_params(trial: optuna.Trial, cfg: TuneConfig) -> dict[str, Any]:
    return dict(
        learning_rate=trial.suggest_float("learning_rate", cfg.lr_min, cfg.lr_max, log=True),
        clip_range=trial.suggest_float("clip_range", cfg.clip_range_min, cfg.clip_range_max),
        ent_coef=trial.suggest_float("ent_coef", cfg.ent_coef_min, cfg.ent_coef_max),
        gamma=trial.suggest_float("gamma", cfg.gamma_min, cfg.gamma_max),
        gae_lambda=trial.suggest_float("gae_lambda", cfg.gae_lambda_min, cfg.gae_lambda_max),
        batch_size=trial.suggest_categorical("batch_size", cfg.batch_size_opts),
        n_epochs=trial.suggest_categorical("n_epochs", cfg.n_epochs_opts),
    )


def _make_objective(tune_cfg: TuneConfig, train_cfg: TrainConfig):
    def objective(trial: optuna.Trial) -> float:
        params = _sample_params(trial, tune_cfg)
        vec_env = DummyVecEnv([lambda: Monitor(ClipRewardWrapper(AtbEnv(train_cfg.config_path)))])
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)
        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            n_steps=train_cfg.ppo.n_steps,
            policy_kwargs=ATB_POLICY_KWARGS,
            verbose=0,
            seed=42,
            device=train_cfg.device,
            **params,
        )
        try:
            model.learn(tune_cfg.n_timesteps)
            mean_reward, _ = evaluate_policy(model, vec_env, n_eval_episodes=3)
        except Exception as exc:
            console.print(f"[red]Trial {trial.number} failed: {exc}[/red]")
            return float("-inf")
        finally:
            vec_env.close()
        return float(mean_reward)

    return objective


@app.command()
def tune(
    stage: int = typer.Option(1, "--stage"),
    n_trials: int = typer.Option(50, "--n-trials"),
    n_timesteps: int = typer.Option(500_000, "--n-timesteps"),
    study_name: str = typer.Option("atb_ppo", "--study-name"),
    device: str = typer.Option("cpu", "--device"),
) -> None:
    tune_cfg = TuneConfig(n_trials=n_trials, n_timesteps=n_timesteps, stage=stage)
    train_cfg = TrainConfig(device=device, stage=stage)

    stats_dir = _RL_ROOT / train_cfg.stats_dir
    stats_dir.mkdir(parents=True, exist_ok=True)
    db_path = stats_dir / f"optuna_{study_name}.db"

    console.rule(f"[bold cyan]Optuna Tuning — {study_name} (stage {stage})")
    console.print(f"  trials     : {n_trials}")
    console.print(f"  timesteps  : {n_timesteps:,} per trial")
    console.print(f"  storage    : {db_path}")

    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{db_path}",
        direction="maximize",
        sampler=TPESampler(seed=42),
        load_if_exists=True,
    )
    study.optimize(_make_objective(tune_cfg, train_cfg), n_trials=n_trials, n_jobs=1)

    console.rule("[bold green]Best Trial")
    console.print(f"  value  : {study.best_value:.4f}")
    console.print(f"  params : {study.best_params}")


if __name__ == "__main__":
    app()