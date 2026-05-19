from __future__ import annotations

from pathlib import Path

import h5py
import pandas as pd


def read_stats(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Stats file not found: {path}")
    with h5py.File(path, "r") as f:
        data = {k: f["episodes"][k][:] for k in f["episodes"].keys()}
    return pd.DataFrame(data).sort_values("timestep").reset_index(drop=True)


def print_summary(path: str | Path) -> None:
    df = read_stats(path)
    print(f"Episodes    : {len(df):,}")
    print(f"Total steps : {df['timestep'].max():,}")
    print(f"Mean reward : {df['episode_reward'].mean():.3f}")
    print(f"Max reward  : {df['episode_reward'].max():.3f}")
    print(f"Score mean  : {df['score'].mean():.2f}")
    print(f"Win rate    : {df['win'].mean() * 100:.1f}%")