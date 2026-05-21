"""Optuna hyperparameter search.

Runs independently of Hydra. Uses the same env factory as train.py so
tuned values transfer cleanly. Pruning via MedianPruner avoids wasting
budget on clearly bad trials.

Usage
-----
    python tune.py --stage 1 --n-trials 50 --n-timesteps 500000
    python tune.py --algo ppo --device cuda --no-pruning
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import optuna
import typer
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from rich.console import Console
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize

from env.factory import build_vec_env
from model.policy import ATB_POLICY_KWARGS
from training.algos import get_algo
from training.config import EnvConfig, PpoConfig, TuneConfig


def linear_schedule(start: float, end: float):
    if start == end:
        return lambda _: float(start)
    return lambda p: float(end + p * (start - end))


console = Console()
app = typer.Typer()

_RL_ROOT = Path(__file__).resolve().parent.parent.parent  # rl/


# ---------------------------------------------------------------------------
# Optuna sampling
# ---------------------------------------------------------------------------

def _sample_ppo(trial: optuna.Trial, cfg: TuneConfig) -> dict[str, Any]:
    return dict(
        learning_rate=trial.suggest_float("learning_rate", cfg.lr_min, cfg.lr_max, log=True),
        clip_range=trial.suggest_float("clip_range", cfg.clip_range_min, cfg.clip_range_max),
        ent_coef=trial.suggest_float("ent_coef", cfg.ent_coef_min, cfg.ent_coef_max),
        gamma=trial.suggest_float("gamma", cfg.gamma_min, cfg.gamma_max),
        gae_lambda=trial.suggest_float("gae_lambda", cfg.gae_lambda_min, cfg.gae_lambda_max),
        batch_size=trial.suggest_categorical("batch_size", cfg.batch_size_opts),
        n_epochs=trial.suggest_categorical("n_epochs", cfg.n_epochs_opts),
    )


# ---------------------------------------------------------------------------
# Pruning callback
# ---------------------------------------------------------------------------

class _PruneCallback(BaseCallback):
    """Report intermediate reward to Optuna and honour pruning signals."""

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


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def _make_objective(tune_cfg: TuneConfig, env_cfg: EnvConfig, algo: str, device: str):
    algo_spec = get_algo(algo)

    def objective(trial: optuna.Trial) -> float:
        params = _sample_ppo(trial, tune_cfg)

        # Build a fresh PpoConfig with sampled values. Fields not sampled
        # (vf_coef, max_grad_norm, n_steps) use the defaults from the YAML.
        # For tuning we build a minimal PpoConfig directly.
        ppo_cfg = PpoConfig(
            n_steps=512,
            batch_size=params["batch_size"],
            n_epochs=params["n_epochs"],
            gamma=params["gamma"],
            gae_lambda=params["gae_lambda"],
            learning_rate=params["learning_rate"],
            learning_rate_final=params["learning_rate"],   # no schedule during tuning
            clip_range=params["clip_range"],
            clip_range_final=params["clip_range"],
            ent_coef=params["ent_coef"],
            ent_coef_final=params["ent_coef"],
            vf_coef=0.5,
            max_grad_norm=0.5,
            target_kl=0.02,
        )

        vec_env = build_vec_env(env_cfg)
        eval_env = build_vec_env(env_cfg, eval_mode=True)

        # Share normalisation stats so eval sees the same scale as training.
        if isinstance(vec_env, VecNormalize) and isinstance(eval_env, VecNormalize):
            eval_env.obs_rms = vec_env.obs_rms
            eval_env.ret_rms = vec_env.ret_rms

        model = algo_spec.constructor(
            policy=algo_spec.policy,
            env=vec_env,
            n_steps=ppo_cfg.n_steps,
            batch_size=ppo_cfg.batch_size,
            n_epochs=ppo_cfg.n_epochs,
            gamma=ppo_cfg.gamma,
            gae_lambda=ppo_cfg.gae_lambda,
            learning_rate=linear_schedule(ppo_cfg.learning_rate, ppo_cfg.learning_rate_final),
            clip_range=ppo_cfg.clip_range,
            ent_coef=ppo_cfg.ent_coef,
            vf_coef=ppo_cfg.vf_coef,
            max_grad_norm=ppo_cfg.max_grad_norm,
            target_kl=ppo_cfg.target_kl,
            policy_kwargs=ATB_POLICY_KWARGS,
            verbose=0,
            seed=42,
            device=device,
        )

        try:
            callbacks = [_PruneCallback(trial)] if tune_cfg.pruning else []
            model.learn(tune_cfg.n_timesteps, callback=callbacks)
            mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=5, deterministic=True)
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.command()
def tune(
    stage: int = typer.Option(1, "--stage"),
    n_trials: int = typer.Option(50, "--n-trials"),
    n_timesteps: int = typer.Option(500_000, "--n-timesteps"),
    n_envs: int = typer.Option(8, "--n-envs"),
    study_name: str = typer.Option("atb_ppo", "--study-name"),
    device: str = typer.Option("cpu", "--device"),
    pruning: bool = typer.Option(True, "--pruning/--no-pruning"),
    algo: str = typer.Option("ppo", "--algo"),
) -> None:
    tune_cfg = TuneConfig(n_trials=n_trials, n_timesteps=n_timesteps, stage=stage, pruning=pruning)

    # Minimal env config for tuning — reward normalisation on, obs normalisation off.
    env_cfg = EnvConfig(
        stage=stage,
        n_envs=n_envs,
        seed=42,
        clip_reward=True,
        clip_reward_max=10.0,
        reward_scale=1.0,
        normalize_obs=False,
        normalize_reward=True,
    )

    stats_dir = _RL_ROOT / "src" / "runs" / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    db_path = stats_dir / f"optuna_{study_name}.db"

    console.rule(f"[bold cyan]Optuna Tuning — {study_name} (stage {stage})")
    console.print(f"  algo       : {algo}")
    console.print(f"  trials     : {n_trials}")
    console.print(f"  timesteps  : {n_timesteps:,} per trial")
    console.print(f"  n_envs     : {n_envs}")
    console.print(f"  pruning    : {pruning}")
    console.print(f"  storage    : {db_path}")

    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=50_000) if pruning else None
    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{db_path}",
        direction="maximize",
        sampler=TPESampler(seed=42),  # type: ignore[arg-type]  # Optuna stub gap
        pruner=pruner,
        load_if_exists=True,
    )
    study.optimize(_make_objective(tune_cfg, env_cfg, algo, device), n_trials=n_trials, n_jobs=1)

    console.rule("[bold green]Best Trial")
    console.print(f"  value  : {study.best_value:.4f}")
    console.print(f"  params : {study.best_params}")


if __name__ == "__main__":
    app()
    