"""EvalCallback subclass that co-saves VecNormalize stats with every best model."""
from __future__ import annotations

from typing import Optional

from stable_baselines3.common.vec_env import VecNormalize

try:
    from sb3_contrib.common.callbacks import MaskableEvalCallback as _EvalBase
except ImportError:
    from stable_baselines3.common.callbacks import EvalCallback as _EvalBase  # type: ignore[assignment]


class EvalWithVecNorm(_EvalBase):
    """EvalCallback that saves vec_normalize alongside every eval-best model.

    SB3's stock EvalCallback writes `best_model.zip` whenever a new best mean
    reward is found but never saves the matching VecNormalize stats. Loading
    that model later (for deployment or resume) silently uses freshly-
    initialised normalisation stats, causing the policy to receive incorrectly
    scaled observations.

    This subclass calls `vec_normalize.save()` every time the parent would
    write a new best model, keeping the pkl in sync.
    """

    def __init__(
            self,
            *args,
            vec_normalize: Optional[VecNormalize] = None,
            **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._vec_normalize = vec_normalize

    def _on_step(self) -> bool:
        result = super()._on_step()
        if self._vec_normalize is not None and self.best_model_save_path is not None:
            vecnorm_path = self.best_model_save_path + "_vecnorm.pkl"
            self._vec_normalize.save(vecnorm_path)
        return result
