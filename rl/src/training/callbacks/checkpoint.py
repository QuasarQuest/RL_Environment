"""Periodic and rolling-best model checkpointing."""
from __future__ import annotations

import copy
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize


class CheckpointCallback(BaseCallback):
    """Save model snapshots periodically and on rolling-best reward.

    The rolling window protects against false "new best" events from a
    single lucky episode. For rigorous best-model tracking use
    `EvalWithVecNorm` instead.

    save_freq is compared against `num_timesteps` (total env steps) rather
    than `n_calls` (number of _on_step invocations). With n_envs=48 one
    _on_step call = 48 steps, so an n_calls check would fire ~48x too often
    relative to the configured timestep budget.

    When onnx_export=True a policy.onnx is written to
    `{run_dir}/models/step_{N}/` alongside every periodic checkpoint so
    checkpoints are immediately usable in the viewer without a separate
    export step.
    """

    def __init__(
            self,
            save_freq: int,
            ckpt_dir: Path,
            vec_normalize: Optional[VecNormalize] = None,
            rolling_window: int = 100,
            verbose: int = 1,
            onnx_export: bool = False,
            min_improvement: float = 2.0,
    ) -> None:
        super().__init__(verbose)
        self.save_freq = save_freq
        self.ckpt_dir = Path(ckpt_dir)
        self.vec_normalize = vec_normalize
        self._onnx_export = onnx_export
        # Only treat a rolling-mean uptick as a "new best" when it clears the
        # previous best by this margin. Without it, the monotonic early-training
        # climb fires a save + log line on every sub-reward improvement, flooding
        # the console and rewriting best_rolling.zip dozens of times per second.
        self._min_improvement = min_improvement
        self._models_dir = self.ckpt_dir.parent / "models"
        self._recent_rewards: deque[float] = deque(maxlen=rolling_window)
        self._best_mean: float = -float("inf")
        self._last_save_step: int = 0
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_save_step >= self.save_freq:
            tag = f"step_{self.num_timesteps}"
            self._save(tag)
            if self._onnx_export:
                self._try_export_onnx(tag)
            self._last_save_step = self.num_timesteps

        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._recent_rewards.append(float(info["episode"]["r"]))

        if len(self._recent_rewards) == self._recent_rewards.maxlen:
            mean_r = float(np.mean(self._recent_rewards))
            if mean_r > self._best_mean + self._min_improvement:
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
        path = self.ckpt_dir / tag
        self.model.save(str(path))
        if self.vec_normalize is not None:
            self.vec_normalize.save(str(path) + "_vecnorm.pkl")

    def _try_export_onnx(self, tag: str) -> None:
        from network.export import export_to_onnx  # noqa: PLC0415

        out_dir = self._models_dir / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = out_dir / "policy.onnx"
        try:
            export_to_onnx(copy.deepcopy(self.model.policy), onnx_path, vecnorm_path=None)
        except Exception as exc:
            print(f"  [ONNX export skipped: {exc}]")
