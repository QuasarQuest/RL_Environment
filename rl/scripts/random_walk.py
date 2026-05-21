"""Random-policy baseline for AtbEnv.

Runs N episodes with a uniformly random action policy and reports pickups,
deliveries, and reward. Use this to sanity-check that the task is solvable
before training — if a random agent never finds gold, PPO won't either.

Usage
-----
    python scripts/random_walk.py
    python scripts/random_walk.py --episodes 50 --seed 1
"""
from __future__ import annotations

from pathlib import Path

import atb
import numpy as np
import typer
from rich import box
from rich.console import Console
from rich.table import Table

app = typer.Typer(add_completion=False)
console = Console()


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
    episodes: int = typer.Option(30, "--episodes"),
    seed: int = typer.Option(0, "--seed"),
) -> None:
    action_size: int = atb.PyRlEnv.action_size()  # type: ignore[attr-defined]
    env = atb.PyRlEnv(_find_config())  # type: ignore[attr-defined]
    rng = np.random.default_rng(seed)

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
        env.reset()
        ep_reward, ep_pickups, ep_deliveries, ticks = 0.0, 0, 0, 0
        done = False

        while not done:
            _, reward, done = env.step(int(rng.integers(0, action_size)))
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

    pickup_rate = np.mean([p > 0 for p in all_pickups])
    console.print()
    if pickup_rate < 0.3:
        console.print("[red]< 30% pickup rate — task may be too sparse for PPO to bootstrap.[/red]")
    elif pickup_rate < 0.7:
        console.print("[yellow]Pickup rate workable but low — training should learn slowly.[/yellow]")
    else:
        console.print("[green]Pickup rate healthy — PPO should learn quickly.[/green]")


if __name__ == "__main__":
    app()
