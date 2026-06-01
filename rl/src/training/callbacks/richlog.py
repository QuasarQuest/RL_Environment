"""Rich console log formatting for SB3 training output."""
from __future__ import annotations

from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.table import Table
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import KVWriter

# Per-table accent colours — title, border and key column share each hue so the
# three panels read as distinct blocks at a glance.
_ACCENT = {
    "time": "cyan",
    "game": "green",
    "train": "magenta",
}


def _fmt(v: object) -> str:
    if isinstance(v, float):
        if abs(v) < 0.001 or abs(v) >= 10_000:
            return f"{v:.3e}"
        return f"{v:.4f}"
    return str(v)


def _band(value: float, *, good: float, warn: float, higher_is_better: bool = True) -> str:
    """Map a metric to green / yellow / red by how it compares to thresholds."""
    if not higher_is_better:
        value = -value
        good, warn = -good, -warn
    if value >= good:
        return "bold green"
    if value >= warn:
        return "yellow"
    return "bold red"


# Fixed colours that don't depend on the value's magnitude.
_STATIC_STYLE = {
    "ep_reward": "bold green", "score": "bold green",
    "loss": "blue", "value_loss": "blue", "policy_loss": "blue", "entropy_loss": "blue",
    "lr": "dim", "ent_coef": "dim",
    "step": "bold white", "iter": "bold white", "elapsed": "bold white",
    "fps": "bold green",
}


def _value_style(label: str, raw: object) -> str:
    """Semantic colour for a metric value; falls back to a neutral bright tone."""
    if label in _STATIC_STYLE:
        return _STATIC_STYLE[label]
    # Threshold-banded metrics need the numeric value (time rows are pre-formatted strings).
    if isinstance(raw, (int, float)):
        v = float(raw)
        if label == "win_rate":
            return "bold green" if v > 0 else "dim"
        if label == "exp_variance":  # 1.0 = perfect value fit, <0 = worse than mean
            return _band(v, good=0.9, warn=0.5)
        if label == "approx_kl":  # target_kl is 0.02 — flag drift past it
            return _band(v, good=0.01, warn=0.02, higher_is_better=False)
        if label == "clip_frac":  # lots of clipping → steps too large
            return _band(v, good=0.1, warn=0.2, higher_is_better=False)
    return "white"


def kv_table(title: str, rows: list[tuple[str, object]]) -> Table:
    accent = _ACCENT.get(title, "cyan")
    t = Table(title=title, box=box.ROUNDED, show_header=False,
              title_style=f"bold {accent}", border_style=f"dim {accent}",
              padding=(0, 1))
    t.add_column(style=accent, no_wrap=True)
    t.add_column(justify="right", no_wrap=True)
    for label, raw in rows:
        t.add_row(label, f"[{_value_style(label, raw)}]{_fmt(raw)}[/]")
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

        # Skip dumps that carry no useful stats:
        # - elapsed == 0 and itr == 0 → eval-callback dump, not the main training loop
        # - train_rows empty (checked below) → partial dump before first rollout completes
        if elapsed == 0 and itr == 0:
            return

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

        # Raw values are kept (not pre-formatted) so kv_table can colour by magnitude.
        time_rows: list[tuple[str, object]] = [
            ("step", f"{steps:,}"),
            ("iter", str(itr)),
            ("fps", f"{fps:,}"),
            ("elapsed", f"{elapsed}s"),
        ]
        game_rows  = [(lbl, kv[key]) for lbl, key in game_keys  if key in kv]
        train_rows = [(lbl, kv[key]) for lbl, key in train_keys if key in kv]

        if not train_rows:
            return  # partial dump before first rollout — superseded by the next full table

        self._console.rule(style="dim")
        tables = [kv_table("time", time_rows)]
        if game_rows:
            tables.append(kv_table("game", game_rows))
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
