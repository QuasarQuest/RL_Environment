"""Evaluate a trained PPO policy against the Rust AtbEnv.

Usage
-----
    python scripts/eval.py --model runs/models/run_s1_XYZ_final
    python scripts/eval.py --model runs/models/run_s1_XYZ_final --episodes 20
    python scripts/eval.py --model runs/models/run_s1_XYZ_final --stochastic
"""
from __future__ import annotations

from pathlib import Path

import atb
import numpy as np
import typer
from rich import box
from rich.console import Console
from rich.table import Table
from stable_baselines3 import PPO

app = typer.Typer(add_completion=False)
console = Console()


def _resolve_model(path: Path) -> Path:
    """Accept a .zip path, a path without extension, or an EvalCallback dir."""
    if path.is_dir():
        candidate = path / "best_model.zip"
        if not candidate.exists():
            raise FileNotFoundError(
                f"{path} is a directory but best_model.zip is not inside it yet. "
                "Has EvalCallback fired? Check eval_freq in your config."
            )
        return candidate
    # PPO.load appends .zip automatically if missing — just pass through
    return path


def _find_config() -> str:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "assets" / "world" / "config.ron"
        if candidate.exists():
            return str(candidate)
    raise RuntimeError(f"Cannot find config.ron from {here}")


def _parse_event(reward: float) -> tuple[int, int]:
    """Return (pickups, deliveries) inferred from a single step reward."""
    if reward > 1.0:
        return 0, max(1, round((reward + 0.001) / 5.0))
    if reward > 0.1:
        return max(1, round((reward + 0.001) / 0.5)), 0
    return 0, 0


@app.command()
def main(
    model_path: Path = typer.Option(..., "--model", help="Path to saved model (.zip)"),
    episodes: int = typer.Option(10, "--episodes", help="Number of eval episodes"),
    deterministic: bool = typer.Option(True, "--deterministic/--stochastic"),
) -> None:
    obs_shape: tuple[int, int, int] = atb.PyRlEnv.obs_shape()  # type: ignore[attr-defined]

    console.print(f"model   : [bold]{model_path}[/bold]")
    console.print(f"mode    : {'deterministic' if deterministic else 'stochastic'}")

    model = PPO.load(str(_resolve_model(model_path)))
    env = atb.PyRlEnv(_find_config())  # type: ignore[attr-defined]

    table = Table(box=box.SIMPLE)
    table.add_column("ep",          justify="right", style="dim")
    table.add_column("reward",      justify="right")
    table.add_column("pickups",     justify="right")
    table.add_column("deliveries",  justify="right")
    table.add_column("ticks",       justify="right")

    all_rewards: list[float] = []
    all_pickups: list[int] = []
    all_deliveries: list[int] = []

    for ep in range(episodes):
        obs = np.asarray(env.reset(), dtype=np.float32).reshape(obs_shape)
        ep_reward, ep_pickups, ep_deliveries, ticks = 0.0, 0, 0, 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs_flat, reward, done = env.step(int(action))
            obs = np.asarray(obs_flat, dtype=np.float32).reshape(obs_shape)
            ep_reward += reward
            p, d = _parse_event(reward)
            ep_pickups += p
            ep_deliveries += d
            ticks += 1

        all_rewards.append(ep_reward)
        all_pickups.append(ep_pickups)
        all_deliveries.append(ep_deliveries)
        table.add_row(str(ep), f"{ep_reward:+.3f}", str(ep_pickups), str(ep_deliveries), str(ticks))

    console.print(table)
    console.print(f"  mean reward     : {np.mean(all_rewards):+.3f} ± {np.std(all_rewards):.3f}")
    console.print(f"  mean pickups    : {np.mean(all_pickups):.2f}")
    console.print(f"  mean deliveries : {np.mean(all_deliveries):.2f}")
    console.print(f"  delivery rate   : {100 * np.mean([d > 0 for d in all_deliveries]):.0f}%")


if __name__ == "__main__":
    app()
