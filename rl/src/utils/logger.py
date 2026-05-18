# src/utils/logger.py
#
# Utility for reading training stats back from HDF5.
# Use this in Jupyter notebooks or analysis scripts.
#
# Example:
#   from src.utils.logger import read_stats
#   df = read_stats("runs/stats/run_123.h5")
#   df["episode_reward"].plot()

from __future__ import annotations
from pathlib import Path
import h5py
import pandas as pd


def read_stats(path: str | Path) -> pd.DataFrame:
    """
    Read all episode stats from an HDF5 file into a Pandas DataFrame.

    Columns:
        timestep, episode_reward, episode_length,
        kills, deaths, score, win
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Stats file not found: {path}")

    with h5py.File(path, "r") as f:
        grp = f["episodes"]
        data = {k: grp[k][:] for k in grp.keys()}

    df = pd.DataFrame(data)
    df = df.sort_values("timestep").reset_index(drop=True)
    return df


def print_summary(path: str | Path) -> None:
    """Print a quick training summary from an HDF5 stats file."""
    df = read_stats(path)
    print(f"Episodes      : {len(df):,}")
    print(f"Total steps   : {df['timestep'].max():,}")
    print(f"Mean reward   : {df['episode_reward'].mean():.3f}")
    print(f"Max reward    : {df['episode_reward'].max():.3f}")
    print(f"Mean kills    : {df['kills'].mean():.2f}")
    print(f"Mean deaths   : {df['deaths'].mean():.2f}")
    print(f"Win rate      : {df['win'].mean()*100:.1f}%")