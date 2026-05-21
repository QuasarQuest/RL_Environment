"""Training callbacks.

Three callbacks live here:

* `CheckpointCallback` — periodic snapshots and "rolling-best" snapshots.
  The "best" is computed over a rolling window of episode rewards rather
  than a single rollout, which prevents a lucky episode from locking in a
  bad model. For evaluation-based best selection, prefer SB3's stock
  `EvalCallback` (wired up in `train.py`).
* `EpisodeStatsCallback` — pulls episode info from Monitor's `infos` and
  fans them out to TensorBoard and the HDF5 stats writer.
* `EntropyCoefScheduleCallback` — actually updates `model.ent_coef`
  each rollout. PPO does not natively schedule entropy; the previous
  implementation passed a callable as `ent_coef` which SB3 stored as-is
  and never invoked, silently freezing the coefficient at construction.
"""
from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.table import Table
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import KVWriter
from stable_baselines3.common.vec_env import VecNormalize

from monitoring.stats_writer import HDF5StatsWriter


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
class CheckpointCallback(BaseCallback):
    """Save model snapshots periodically and on rolling-best reward.

    The rolling window protects against false "new best" events from a
    single lucky episode. For rigorous best-model tracking use
    `EvalCallback` instead.
    """

    def __init__(
        self,
        save_freq: int,
        models_dir: Path,
        run_name: str,
        vec_normalize: Optional[VecNormalize] = None,
        rolling_window: int = 100,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.save_freq = save_freq
        self.models_dir = Path(models_dir)
        self.run_name = run_name
        self.vec_normalize = vec_normalize
        self._recent_rewards: deque[float] = deque(maxlen=rolling_window)
        self._best_mean: float = -float("inf")
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        # Periodic snapshot.
        if self.n_calls % self.save_freq == 0:
            self._save(f"step_{self.num_timesteps}")

        # Track rolling reward. We need enough samples for "best" to be
        # meaningful, so we gate on a full window.
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._recent_rewards.append(float(info["episode"]["r"]))

        if len(self._recent_rewards) == self._recent_rewards.maxlen:
            mean_r = float(np.mean(self._recent_rewards))
            if mean_r > self._best_mean:
                self._best_mean = mean_r
                self._save("best_rolling")
                if self.verbose:
                    print(
                        f"  ✓ New rolling-best mean reward {mean_r:.3f} "
                        f"(window={self._recent_rewards.maxlen}) @ step "
                        f"{self.num_timesteps}"
                    )
        return True

    def _save(self, tag: str) -> None:
        path = self.models_dir / f"{self.run_name}_{tag}"
        self.model.save(str(path))
        if self.vec_normalize is not None:
            self.vec_normalize.save(str(path) + "_vecnorm.pkl")


# ---------------------------------------------------------------------------
# Episode stats → TensorBoard + HDF5
# ---------------------------------------------------------------------------
class EpisodeStatsCallback(BaseCallback):
    """Record per-episode stats to TensorBoard and an HDF5 file."""

    def __init__(self, stats_path: Path, flush_every: int = 100, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._writer = HDF5StatsWriter(Path(stats_path), flush_every=flush_every)
        self._start_time = time.time()

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" not in info:
                continue
            ep = info["episode"]
            row = {
                "timestep": self.num_timesteps,
                "episode_reward": float(ep["r"]),
                "episode_length": int(ep["l"]),
                "score": float(info.get("score", 0.0)),
                "win": int(info.get("win", 0)),
            }
            self._writer.append(row)

            self.logger.record("game/episode_reward", row["episode_reward"])
            self.logger.record("game/episode_length", row["episode_length"])
            self.logger.record("game/score", row["score"])
            self.logger.record("game/win_rate", row["win"])

        elapsed = max(time.time() - self._start_time, 1.0)
        self.logger.record("perf/steps_per_sec", self.num_timesteps / elapsed)
        return True

    def _on_training_end(self) -> None:
        self._writer.close()


# ---------------------------------------------------------------------------
# Entropy coefficient schedule
# ---------------------------------------------------------------------------
class EntropyCoefScheduleCallback(BaseCallback):
    """Drive `model.ent_coef` along a schedule.

    PPO's `ent_coef` is a plain float read inside the loss; SB3 ignores any
    callable passed there. This callback updates the attribute at the start
    of each rollout, mirroring how `learning_rate` and `clip_range` are
    natively handled.
    """

    def __init__(
        self,
        schedule: Callable[[float], float],
        total_timesteps: int,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self._schedule = schedule
        self._total = max(int(total_timesteps), 1)

    def _on_rollout_start(self) -> None:
        progress_remaining = 1.0 - (self.num_timesteps / self._total)
        progress_remaining = max(0.0, min(1.0, progress_remaining))
        new_val = float(self._schedule(progress_remaining))
        self.model.ent_coef = new_val
        self.logger.record("train/ent_coef", new_val)

    def _on_step(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Rich log output
# ---------------------------------------------------------------------------

def _fmt(v: object) -> str:
    if isinstance(v, float):
        if abs(v) < 0.001 or abs(v) >= 10_000:
            return f"{v:.3e}"
        return f"{v:.4f}"
    return str(v)


def _kv(title: str, rows: list[tuple[str, str]]) -> Table:
    t = Table(title=title, box=box.SIMPLE, show_header=False,
              title_style="bold cyan", padding=(0, 1))
    t.add_column(style="dim", no_wrap=True)
    t.add_column(justify="right", no_wrap=True)
    for k, v in rows:
        t.add_row(k, v)
    return t


class _RichWriter(KVWriter):
    """Formats SB3's key-value log dumps as a compact Rich table."""

    def __init__(self, console: Console) -> None:
        self._console = console

    def write(self, key_values: dict, key_excluded: dict, step: int = 0) -> None:
        # Don't apply key_excluded — _RichWriter is neither "stdout" nor
        # "tensorboard", so exclusions meant for those formats apply to them,
        # not here.  time/iterations, time/time_elapsed, etc. are excluded from
        # tensorboard in SB3 2.8+ and would otherwise all show as 0.
        kv = dict(key_values)

        steps   = int(kv.get("time/total_timesteps", step))
        itr     = int(kv.get("time/iterations", 0))
        fps     = int(kv.get("time/fps", 0))
        elapsed = int(kv.get("time/time_elapsed", 0))

        # elapsed == 0 means this dump came from EvalCallback, not the main
        # training loop — time/* keys are absent. Skip the separator so eval
        # output flows inline without a misleading all-zero time table.
        is_eval_dump = elapsed == 0 and itr == 0
        if not is_eval_dump:
            self._console.rule(style="dim")

        game_keys = [
            ("ep_reward",  "game/episode_reward"),
            ("ep_length",  "game/episode_length"),
            ("score",      "game/score"),
            ("win_rate",   "game/win_rate"),
        ]
        train_keys = [
            ("loss",          "train/loss"),
            ("value_loss",    "train/value_loss"),
            ("policy_loss",   "train/policy_gradient_loss"),
            ("entropy_loss",  "train/entropy_loss"),
            ("approx_kl",     "train/approx_kl"),
            ("clip_frac",     "train/clip_fraction"),
            ("exp_variance",  "train/explained_variance"),
            ("lr",            "train/learning_rate"),
            ("ent_coef",      "train/ent_coef"),
        ]

        time_rows = [
            ("step",    f"{steps:,}"),
            ("iter",    str(itr)),
            ("fps",     f"{fps:,}"),
            ("elapsed", f"{elapsed}s"),
        ]
        game_rows = [(lbl, _fmt(kv[key])) for lbl, key in game_keys if key in kv]
        train_rows = [(lbl, _fmt(kv[key])) for lbl, key in train_keys if key in kv]

        tables = []
        if not is_eval_dump:
            tables.append(_kv("time", time_rows))
        if game_rows:
            tables.append(_kv("game", game_rows))
        if train_rows:
            tables.append(_kv("train", train_rows))
        if tables:
            self._console.print(Columns(tables, equal=False, expand=False))

    def close(self) -> None:
        pass


class RichLogCallback(BaseCallback):
    """Injects _RichWriter into SB3's logger on training start.

    Use with verbose=0 on the model so SB3 doesn't also add its own
    HumanOutputFormat to stdout.
    """

    def __init__(self, console: Console) -> None:
        super().__init__()
        self._writer = _RichWriter(console)

    def _on_training_start(self) -> None:
        self.model.logger.output_formats.append(self._writer)

    def _on_step(self) -> bool:
        return True
