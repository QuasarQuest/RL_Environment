"""Optuna hyperparameter search.

Shares the env factory with `train.py` so tuned values transfer cleanly.
Adds a MedianPruner with intermediate reporting to avoid spending budget
on obviously-bad trials.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import optuna
import typer
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from rich.console import Console
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize

from env.factory import build_vec_env
from model.policy import ATB_POLICY_KWARGS
from training.algos import get_algo
from training.config import PpoConfig, TrainConfig, TuneConfig

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


class _OptunaReportCallback(BaseCallback):
    """Report intermediate mean rollout reward to Optuna and honour pruning."""

    def __init__(self, trial: optuna.Trial, report_every: int = 10_000) -> None:
        super().__init__(verbose=0)
        self.trial = trial
        self.report_every = report_every
        self._last_report = 0
        self._recent: list[float] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._recent.append(float(info["episode"]["r"]))

        if self.num_timesteps - self._last_report >= self.report_every and self._recent:
            mean_r = sum(self._recent) / len(self._recent)
            self.trial.report(mean_r, step=self.num_timesteps)
            self._recent.clear()
            self._last_report = self.num_timesteps
            if self.trial.should_prune():
                raise optuna.TrialPruned()
        return True


def _make_objective(tune_cfg: TuneConfig, train_cfg: TrainConfig):
    algo_spec = get_algo(train_cfg.algo)

    def objective(trial: optuna.Trial) -> float:
        params = _sample_params(trial, tune_cfg)

        # Build a per-trial config so the env factory produces the exact
        # same wrapper stack as during training.
        trial_cfg = dataclasses.replace(
            train_cfg,
            ppo=PpoConfig(
                **{**dataclasses.asdict(train_cfg.ppo),
                   "batch_size": params["batch_size"],
                   "n_epochs": params["n_epochs"],
                   "gamma": params["gamma"],
                   "gae_lambda": params["gae_lambda"]},
            ),
        )

        vec_env = build_vec_env(trial_cfg)
        eval_env = build_vec_env(trial_cfg, eval_mode=True)
        if isinstance(vec_env, VecNormalize) and isinstance(eval_env, VecNormalize):
            eval_env.obs_rms = vec_env.obs_rms
            eval_env.ret_rms = vec_env.ret_rms

        model = algo_spec.cls(
            policy=algo_spec.policy,
            env=vec_env,
            n_steps=trial_cfg.ppo.n_steps,
            batch_size=params["batch_size"],
            n_epochs=params["n_epochs"],
            gamma=params["gamma"],
            gae_lambda=params["gae_lambda"],
            learning_rate=params["learning_rate"],
            clip_range=params["clip_range"],
            ent_coef=params["ent_coef"],
            vf_coef=trial_cfg.ppo.vf_coef,
            max_grad_norm=trial_cfg.ppo.max_grad_norm,
            policy_kwargs=ATB_POLICY_KWARGS,
            verbose=0,
            seed=42,
            device=trial_cfg.device,
        )

        try:
            callbacks = [_OptunaReportCallback(trial)] if tune_cfg.pruning else []
            model.learn(tune_cfg.n_timesteps, callback=callbacks)
            mean_reward, _ = evaluate_policy(
                model, eval_env, n_eval_episodes=5, deterministic=True
            )
        except optuna.TrialPruned:
            raise
        except Exception as exc:
            console.print(f"[red]Trial {trial.number} failed: {exc}[/red]")
            return float("-inf")
        finally:
            vec_env.close()
            eval_env.close()

        return float(mean_reward)

    return objective


@app.command()
def tune(
    stage: int = typer.Option(1, "--stage"),
    n_trials: int = typer.Option(50, "--n-trials"),
    n_timesteps: int = typer.Option(500_000, "--n-timesteps"),
    study_name: str = typer.Option("atb_ppo", "--study-name"),
    device: str = typer.Option("cpu", "--device"),
    pruning: bool = typer.Option(True, "--pruning/--no-pruning"),
    algo: str = typer.Option("ppo", "--algo"),
) -> None:
    tune_cfg = TuneConfig(
        n_trials=n_trials,
        n_timesteps=n_timesteps,
        stage=stage,
        pruning=pruning,
    )
    train_cfg = TrainConfig(device=device, stage=stage, algo=algo)

    stats_dir = _RL_ROOT / train_cfg.stats_dir
    stats_dir.mkdir(parents=True, exist_ok=True)
    db_path = stats_dir / f"optuna_{study_name}.db"

    console.rule(f"[bold cyan]Optuna Tuning — {study_name} (stage {stage})")
    console.print(f"  algo       : {algo}")
    console.print(f"  trials     : {n_trials}")
    console.print(f"  timesteps  : {n_timesteps:,} per trial")
    console.print(f"  pruning    : {pruning}")
    console.print(f"  storage    : {db_path}")

    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=50_000) if pruning else None
    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{db_path}",
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=pruner,
        load_if_exists=True,
    )
    study.optimize(_make_objective(tune_cfg, train_cfg), n_trials=n_trials, n_jobs=1)

    console.rule("[bold green]Best Trial")
    console.print(f"  value  : {study.best_value:.4f}")
    console.print(f"  params : {study.best_params}")


if __name__ == "__main__":
    app()
