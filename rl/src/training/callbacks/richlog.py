"""Rich console log formatting for SB3 training output."""
from __future__ import annotations

from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.table import Table
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import KVWriter


def _fmt(v: object) -> str:
    if isinstance(v, float):
        if abs(v) < 0.001 or abs(v) >= 10_000:
            return f"{v:.3e}"
        return f"{v:.4f}"
    return str(v)


def kv_table(title: str, rows: list[tuple[str, str]]) -> Table:
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
        kv = dict(key_values)

        steps = int(kv.get("time/total_timesteps", step))
        itr = int(kv.get("time/iterations", 0))
        fps = int(kv.get("time/fps", 0))
        elapsed = int(kv.get("time/time_elapsed", 0))

        # elapsed == 0 means this dump came from EvalCallback, not the main
        # training loop. Skip to avoid duplicating eval metrics in the table.
        if elapsed == 0 and itr == 0:
            return

        self._console.rule(style="dim")

        game_keys = [
            ("ep_reward", "game/episode_reward"),
            ("ep_length", "game/episode_length"),
            ("score", "game/score"),
            ("win_rate", "game/win_rate"),
        ]
        train_keys = [
            ("loss", "train/loss"),
            ("value_loss", "train/value_loss"),
            ("policy_loss", "train/policy_gradient_loss"),
            ("entropy_loss", "train/entropy_loss"),
            ("approx_kl", "train/approx_kl"),
            ("clip_frac", "train/clip_fraction"),
            ("exp_variance", "train/explained_variance"),
            ("lr", "train/learning_rate"),
            ("ent_coef", "train/ent_coef"),
        ]

        time_rows = [
            ("step", f"{steps:,}"),
            ("iter", str(itr)),
            ("fps", f"{fps:,}"),
            ("elapsed", f"{elapsed}s"),
        ]
        game_rows = [(lbl, _fmt(kv[key])) for lbl, key in game_keys if key in kv]
        train_rows = [(lbl, _fmt(kv[key])) for lbl, key in train_keys if key in kv]

        tables = [kv_table("time", time_rows)]
        if game_rows:
            tables.append(kv_table("game", game_rows))
        if train_rows:
            tables.append(kv_table("train", train_rows))
        self._console.print(Columns(tables, equal=False, expand=False))

    def close(self) -> None:
        pass


class RichLogCallback(BaseCallback):
    """Injects _RichWriter into SB3's logger on training start."""

    def __init__(self, console: Console) -> None:
        super().__init__()
        self._writer = _RichWriter(console)

    def _on_training_start(self) -> None:
        self.model.logger.output_formats.append(self._writer)

    def _on_step(self) -> bool:
        return True
