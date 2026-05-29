"""EvalCallback subclass that co-saves VecNormalize stats with every best model."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional

from stable_baselines3.common.vec_env import VecNormalize

try:
    from sb3_contrib.common.callbacks import MaskableEvalCallback as _EvalBase
except ImportError:
    from stable_baselines3.common.callbacks import EvalCallback as _EvalBase  # type: ignore[assignment]


class EvalWithVecNorm(_EvalBase):
    """EvalCallback that saves VecNormalize stats and exports ONNX on every new eval-best.

    SB3's stock EvalCallback writes `best_model.zip` on a new best mean reward
    but never saves the matching VecNormalize stats, so a resume or deployment
    load silently uses fresh (mean=0, var=1) statistics.
    """

    def __init__(
            self,
            *args,
            vec_normalize: Optional[VecNormalize] = None,
            onnx_path: Optional[Path] = None,
            **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._vec_normalize = vec_normalize
        self._onnx_path = Path(onnx_path) if onnx_path else None

    def _on_step(self) -> bool:
        prev_best = self.best_mean_reward
        result    = super()._on_step()

        if self.best_mean_reward <= prev_best:
            return result

        if self._vec_normalize is not None and self.best_model_save_path is not None:
            self._vec_normalize.save(self.best_model_save_path + "_vecnorm.pkl")

        if self._onnx_path is not None:
            self._try_export_onnx()

        return result

    def _try_export_onnx(self) -> None:
        from network.export import export_to_onnx  # noqa: PLC0415

        vn_path = (
            Path(self.best_model_save_path + "_vecnorm.pkl")
            if self.best_model_save_path else None
        )
        try:
            export_to_onnx(
                copy.deepcopy(self.model.policy),
                self._onnx_path,
                vecnorm_path=vn_path if vn_path and vn_path.exists() else None,
            )
        except Exception as exc:
            print(f"  [ONNX export skipped: {exc}]")
