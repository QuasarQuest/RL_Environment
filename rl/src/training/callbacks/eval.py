"""EvalCallback subclass that co-saves VecNormalize stats with every best model."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from stable_baselines3.common.vec_env import VecNormalize

try:
    from sb3_contrib.common.callbacks import MaskableEvalCallback as _EvalBase
except ImportError:
    from stable_baselines3.common.callbacks import EvalCallback as _EvalBase  # type: ignore[assignment]


class EvalWithVecNorm(_EvalBase):
    """EvalCallback that saves vec_normalize alongside every eval-best model
    and optionally exports the best policy to ONNX for the viewer.

    SB3's stock EvalCallback writes `best_model.zip` whenever a new best mean
    reward is found but never saves the matching VecNormalize stats. Loading
    that model later (for deployment or resume) silently uses freshly-
    initialised normalisation stats, causing the policy to receive incorrectly
    scaled observations.

    When `onnx_path` is supplied, a fresh ONNX export is written every time
    a new eval-best is reached so the viewer always reflects the latest
    best policy without waiting for training to finish.
    """

    def __init__(
            self,
            *args,
            vec_normalize: Optional[VecNormalize] = None,
            onnx_path: Optional[Path] = None,
            viewer_onnx_path: Optional[Path] = None,
            **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._vec_normalize    = vec_normalize
        self._onnx_path        = Path(onnx_path)        if onnx_path        else None
        self._viewer_onnx_path = Path(viewer_onnx_path) if viewer_onnx_path else None

    def _on_step(self) -> bool:
        prev_best = self.best_mean_reward
        result    = super()._on_step()

        if self.best_mean_reward <= prev_best:
            return result  # no new best — nothing extra to do

        # ── New eval-best ──────────────────────────────────────────────────────

        if self._vec_normalize is not None and self.best_model_save_path is not None:
            vecnorm_path = self.best_model_save_path + "_vecnorm.pkl"
            self._vec_normalize.save(vecnorm_path)

        if self._onnx_path is not None:
            self._try_export_onnx()

        return result

    def _try_export_onnx(self) -> None:
        import copy
        from network.export import export_to_onnx

        onnx_path = self._onnx_path
        if onnx_path is None:
            return
        try:
            vn_path = (
                Path(self.best_model_save_path + "_vecnorm.pkl")
                if self.best_model_save_path else None
            )
            export_to_onnx(
                copy.deepcopy(self.model.policy),
                onnx_path,
                vecnorm_path=vn_path if vn_path and vn_path.exists() else None,
            )
            if self._viewer_onnx_path is not None:
                shutil.copy2(onnx_path, self._viewer_onnx_path)
        except Exception as exc:
            print(f"  [ONNX export skipped: {exc}]")
