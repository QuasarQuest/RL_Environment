"""Eval-best callbacks that co-save VecNormalize stats and export ONNX.

`make_eval_callback` selects the correct eval base for the algorithm:
maskable_ppo needs sb3-contrib's MaskableEvalCallback (it fetches action masks
and passes them into predict); plain ppo must use the stock EvalCallback. Using
the maskable base with a non-maskable model would pass an `action_masks` kwarg its
predict() rejects and crash evaluation — so the base is chosen by algo, not by
import availability.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import TYPE_CHECKING

from stable_baselines3.common.callbacks import EvalCallback

try:
    from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
    _MASKABLE_AVAILABLE = True
except ImportError:
    MaskableEvalCallback = None  # type: ignore[assignment,misc]
    _MASKABLE_AVAILABLE = False


# At runtime the mixin is combined with a concrete EvalCallback subclass (see
# EvalWithVecNorm / MaskableEvalWithVecNorm), inheriting best_mean_reward,
# best_model_save_path, model and _on_step from it. Declaring EvalCallback as the
# static base (type-checking only) lets the checker resolve those attributes
# without changing the runtime MRO — the real base is supplied by the subclass.
if TYPE_CHECKING:
    _EvalBase = EvalCallback
else:
    _EvalBase = object


class _VecNormOnnxEvalMixin(_EvalBase):
    """On every new eval-best: co-save VecNormalize stats and export ONNX.

    SB3's stock EvalCallback writes only `best_model.zip` and never the matching
    VecNormalize stats, so a resume or deployment load silently uses fresh
    (mean=0, var=1) statistics. Mix this into the correct eval base for the algo.
    """

    def __init__(self, *args, vec_normalize=None, onnx_path=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._vec_normalize = vec_normalize
        self._onnx_path = Path(onnx_path) if onnx_path else None

    def _on_step(self) -> bool:
        prev_best = self.best_mean_reward
        result = super()._on_step()

        if self.best_mean_reward <= prev_best:
            return result

        if self._vec_normalize is not None and self.best_model_save_path is not None:
            self._vec_normalize.save(self.best_model_save_path + "_vecnorm.pkl")

        if self._onnx_path is not None:
            self._try_export_onnx()

        return result

    def _try_export_onnx(self) -> None:
        from network.export import export_to_onnx  # noqa: PLC0415

        assert self._onnx_path is not None  # only called from _on_step under this guard
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


class EvalWithVecNorm(_VecNormOnnxEvalMixin, EvalCallback):
    """Standard (non-masking) eval-best callback with VecNorm + ONNX co-save."""


if _MASKABLE_AVAILABLE:

    class MaskableEvalWithVecNorm(_VecNormOnnxEvalMixin, MaskableEvalCallback):
        """Maskable eval-best callback — fetches action masks during evaluation."""


def make_eval_callback(algo: str, *args, **kwargs):
    """Return the eval-best callback matching the algorithm's masking support."""
    if algo == "maskable_ppo":
        if not _MASKABLE_AVAILABLE:
            raise RuntimeError(
                "algo=maskable_ppo requires sb3-contrib (MaskableEvalCallback)."
            )
        return MaskableEvalWithVecNorm(*args, **kwargs)
    return EvalWithVecNorm(*args, **kwargs)
