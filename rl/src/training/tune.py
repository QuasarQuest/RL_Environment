# src/training/tune.py
#
# Optuna hyperparameter tuning for PPO.
#
# Usage:
#   python -m src.training.tune --n-trials 50 --n-timesteps 500000
#   atb-tune --n-trials 50
#
# Each trial trains for n_timesteps and reports mean episode reward.
# Results are saved to runs/stats/optuna_<study_name>.db (SQLite).
# Resume an interrupted study by passing the same --study-name.

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import optuna
import typer
from optuna.samplers import TPESampler
from rich.console import Console
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

from src.env.atb_env import AtbEnv
from src.env.wrappers import ClipRewardWrapper
from src.model.policy import ATB_POLICY_KWARGS
from src.training.config import TrainConfig, TuneConfig

console = Console()
app     = typer.Typer()


def sample_ppo_params(trial: optuna.Trial, tune_cfg: TuneConfig) -> dict[str, Any]:
    """Sample PPO hyperparameters from the search space."""
    return dict(
        learning_rate = trial.suggest_float(
            "learning_rate", tune_cfg.lr_min, tune_cfg.lr_max, log=True),
        clip_range    = trial.suggest_float(
            "clip_range", tune_cfg.clip_range_min, tune_cfg.clip_range_max),
        ent_coef      = trial.suggest_float(
            "ent_coef", tune_cfg.ent_coef_min, tune_cfg.ent_coef_max),
        gamma         = trial.suggest_float(
            "gamma", tune_cfg.gamma_min, tune_cfg.gamma_max),
        gae_lambda    = trial.suggest_float(
            "gae_lambda", tune_cfg.gae_lambda_min, tune_cfg.gae_lambda_max),
        batch_size    = trial.suggest_categorical(
            "batch_size", tune_cfg.batch_size_opts),
        n_epochs      = trial.suggest_categorical(
            "n_epochs", tune_cfg.n_epochs_opts),
    )


def make_objective(tune_cfg: TuneConfig, train_cfg: TrainConfig):
    """Returns an Optuna objective function."""

    def objective(trial: optuna.Trial) -> float:
        params = sample_ppo_params(trial, tune_cfg)

        vec_env = DummyVecEnv([lambda: Monitor(ClipRewardWrapper(AtbEnv()))])
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)

        model = PPO(
            policy        = "MlpPolicy",
            env           = vec_env,
            n_steps       = 10_000,
            policy_kwargs = ATB_POLICY_KWARGS,
            verbose       = 0,
            seed          = 42,
            device        = train_cfg.device,
            **params,
        )

        try:
            model.learn(tune_cfg.n_timesteps)
            mean_reward, _ = evaluate_policy(model, vec_env, n_eval_episodes=3)
        except Exception as e:
            console.print(f"[red]Trial {trial.number} failed: {e}[/red]")
            return float("-inf")
        finally:
            vec_env.close()

        return float(mean_reward)

    return objective


@app.command()
def tune(
    n_trials:      int = typer.Option(50,        "--n-trials"),
    n_timesteps:   int = typer.Option(1_000_000, "--n-timesteps"),
    study_name:    str = typer.Option("atb_ppo", "--study-name"),
    device:        str = typer.Option("auto",    "--device"),
):
    """Run Optuna hyperparameter search for PPO."""

    tune_cfg  = TuneConfig(n_trials=n_trials, n_timesteps=n_timesteps)
    train_cfg = TrainConfig(device=device)

    stats_dir = Path(train_cfg.stats_dir)
    stats_dir.mkdir(parents=True, exist_ok=True)
    db_path   = stats_dir / f"optuna_{study_name}.db"

    console.rule(f"[bold cyan]Optuna Tuning — {study_name}")
    console.print(f"  trials      : {n_trials}")
    console.print(f"  timesteps   : {n_timesteps:,} per trial")
    console.print(f"  storage     : {db_path}")

    study = optuna.create_study(
        study_name  = study_name,
        storage     = f"sqlite:///{db_path}",
        direction   = "maximize",
        sampler     = TPESampler(seed=42),
        load_if_exists = True,
    )

    study.optimize(
        make_objective(tune_cfg, train_cfg),
        n_trials  = n_trials,
        n_jobs    = 1,   # one env per process — no parallelism here
        show_progress_bar = True,
    )

    console.rule("[bold green]Best Trial")
    console.print(f"  value  : {study.best_value:.4f}")
    console.print(f"  params : {study.best_params}")


if __name__ == "__main__":
    app()