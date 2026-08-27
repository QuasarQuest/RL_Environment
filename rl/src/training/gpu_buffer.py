"""GPU-resident minibatch sampling for on-policy rollout buffers.

SB3's default RolloutBuffer.get()/_get_samples() calls self.to_torch() once per
field per minibatch — each call is its own torch.tensor(..., device=cuda), i.e. its
own CPU->GPU transfer. Measured at ~9,000 individual torch.tensor() calls (~17% of
wall time) in a profiled run of this project's config (7 fields x 16 minibatches x
4 epochs x 10 rollouts). Worse: since `get()` is called once per PPO epoch and
generator_ready only guards the swap_and_flatten reshape (not the per-minibatch
to_torch calls), stock SB3 re-transfers every field from scratch every epoch too.

This mixin transfers each field to GPU exactly once per rollout — right after
swap_and_flatten, gated by the same `generator_ready` flag — and indexes
minibatches with GPU-resident tensors afterward, including across epochs.
Mirrors the upstream get()/_get_samples() minibatch semantics (same permutation
call pattern, same batch boundaries) exactly; only transfer granularity changes.
"""
from __future__ import annotations

from collections.abc import Generator
from typing import TYPE_CHECKING

import numpy as np
import torch as th
from sb3_contrib.common.maskable.buffers import MaskableRolloutBufferSamples
from stable_baselines3.common.type_aliases import RolloutBufferSamples

if TYPE_CHECKING:
    from stable_baselines3.common.buffers import RolloutBuffer as _BufBase
else:
    _BufBase = object


class _GpuSampleMixin(_BufBase):
    """Shared get() — subclasses set `_tensor_fields` and `_make_samples`."""

    _tensor_fields: tuple[str, ...]
    _gpu_fields: dict[str, th.Tensor]

    #: Fields stored as (buffer_size, n_envs) — i.e. one scalar per sample.
    #: swap_and_flatten() pads 2D input to a trailing size-1 dim (see its
    #: docstring/impl: `shape = (*shape, 1)` when len(shape) < 3), giving
    #: (buffer_len, 1). Stock _get_samples squeezes that via `.flatten()` per
    #: minibatch; we do it once, on the full array, right after flattening.
    _scalar_fields = ("values", "log_probs", "advantages", "returns")

    def get(self, batch_size: int | None = None) -> Generator:
        assert self.full, ""
        buffer_len = self.buffer_size * self.n_envs
        indices = np.random.permutation(buffer_len)

        if not self.generator_ready:
            for name in self._tensor_fields:
                flat = self.swap_and_flatten(getattr(self, name))
                if name in self._scalar_fields:
                    flat = flat.reshape(-1)
                setattr(self, name, flat)
            self.generator_ready = True
            # One H2D transfer per field per rollout — reused across all n_epochs
            # of get() calls until the next reset().
            self._gpu_fields = {name: self.to_torch(getattr(self, name)) for name in self._tensor_fields}

        if batch_size is None:
            batch_size = buffer_len

        idx_t = th.as_tensor(indices, device=self.device)
        start_idx = 0
        while start_idx < buffer_len:
            yield self._make_samples(idx_t[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _make_samples(self, batch_idx: th.Tensor):
        raise NotImplementedError


class GpuPpoSampleMixin(_GpuSampleMixin):
    """Plain-PPO field set (no action_masks)."""

    _tensor_fields = ("observations", "actions", "values", "log_probs", "advantages", "returns")

    def _make_samples(self, batch_idx: th.Tensor) -> RolloutBufferSamples:
        g = self._gpu_fields
        return RolloutBufferSamples(
            observations=g["observations"][batch_idx],
            actions=g["actions"][batch_idx],
            old_values=g["values"][batch_idx],
            old_log_prob=g["log_probs"][batch_idx],
            advantages=g["advantages"][batch_idx],
            returns=g["returns"][batch_idx],
        )


class GpuMaskableSampleMixin(_GpuSampleMixin):
    """MaskablePPO field set (adds action_masks)."""

    _tensor_fields = (
        "observations", "actions", "values", "log_probs", "advantages", "returns", "action_masks",
    )

    def _make_samples(self, batch_idx: th.Tensor) -> MaskableRolloutBufferSamples:
        g = self._gpu_fields
        return MaskableRolloutBufferSamples(
            observations=g["observations"][batch_idx],
            actions=g["actions"][batch_idx],
            old_values=g["values"][batch_idx],
            old_log_prob=g["log_probs"][batch_idx],
            advantages=g["advantages"][batch_idx],
            returns=g["returns"][batch_idx],
            action_masks=g["action_masks"][batch_idx],
        )
