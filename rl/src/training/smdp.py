"""Semi-MDP (option) discounting for PPO / MaskablePPO.

The env runs temporally-extended options (SimCore::step_option): one agent step is
one high-level decision spanning k sim ticks. The option's OWN reward is already the
intra-option discounted sum Σ γ^i r_i (computed in Rust). What standard PPO/GAE gets
wrong is the CROSS-option bootstrap — it discounts the next state's value by a fixed
γ per decision, regardless of how long the option took. The correct semi-MDP rule is
γ^k for an option of length k, so a long trek to far gold is properly devalued
relative to a short hop to near gold (and a far detour only wins when its payoff
justifies the γ^k discount).

This module provides:
  - a RolloutBuffer that applies γ^(option_ticks) in GAE, and
  - a callback that feeds it the per-step option lengths from the env.

Wire both in train.py:
    rb_cls = smdp_buffer_class(algo)                  # None for unsupported algos
    model  = Algo(..., rollout_buffer_class=rb_cls)
    callbacks += [SmdpDiscountCallback()]

recurrent_ppo uses RecurrentRolloutBuffer and is not covered here; with it the
callback simply no-ops (the buffer has no `option_ticks`), falling back to plain γ.
"""
from __future__ import annotations

import numpy as np
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback

try:
    from sb3_contrib.common.maskable.buffers import MaskableRolloutBuffer
    _MASKABLE = True
except ImportError:
    MaskableRolloutBuffer = None  # type: ignore[assignment,misc]
    _MASKABLE = False


class _SmdpGaeMixin:
    """Override GAE to discount each transition by γ^(option_ticks) instead of γ.

    Mirrors stable_baselines3.common.buffers.RolloutBuffer.compute_returns_and_advantage
    exactly, with the single change that the scalar ``self.gamma`` is replaced by a
    per-env ``self.gamma ** self.option_ticks[step]``.
    """

    def reset(self) -> None:
        super().reset()  # type: ignore[misc]
        # Default 1 → γ^1 = γ (plain PPO) until the callback writes real option lengths.
        self.option_ticks = np.ones((self.buffer_size, self.n_envs), dtype=np.float32)  # type: ignore[attr-defined]

    def compute_returns_and_advantage(self, last_values, dones) -> None:  # type: ignore[override]
        last_values = last_values.clone().cpu().numpy().flatten()
        last_gae_lam = 0.0
        for step in reversed(range(self.buffer_size)):  # type: ignore[attr-defined]
            if step == self.buffer_size - 1:  # type: ignore[attr-defined]
                next_non_terminal = 1.0 - dones.astype(np.float32)
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]  # type: ignore[attr-defined]
                next_values = self.values[step + 1]  # type: ignore[attr-defined]
            disc = self.gamma ** self.option_ticks[step]  # γ^k, per env  # type: ignore[attr-defined]
            delta = self.rewards[step] + disc * next_values * next_non_terminal - self.values[step]  # type: ignore[attr-defined]
            last_gae_lam = delta + disc * self.gae_lambda * next_non_terminal * last_gae_lam  # type: ignore[attr-defined]
            self.advantages[step] = last_gae_lam  # type: ignore[attr-defined]
        self.returns = self.advantages + self.values  # type: ignore[attr-defined]


class SmdpRolloutBuffer(_SmdpGaeMixin, RolloutBuffer):
    """SMDP-discounted rollout buffer for plain PPO."""


if _MASKABLE:

    class SmdpMaskableRolloutBuffer(_SmdpGaeMixin, MaskableRolloutBuffer):
        """SMDP-discounted rollout buffer for MaskablePPO."""


def smdp_buffer_class(algo: str):
    """Return the SMDP buffer class for `algo`, or None if unsupported."""
    if algo == "maskable_ppo":
        return SmdpMaskableRolloutBuffer if _MASKABLE else None
    if algo == "ppo":
        return SmdpRolloutBuffer
    return None  # recurrent_ppo: plain γ (RecurrentRolloutBuffer not subclassed)


class SmdpDiscountCallback(BaseCallback):
    """Feed per-step option lengths from the env into the SMDP rollout buffer.

    SB3 calls callback.on_step() once per env step, BEFORE rollout_buffer.add, so
    writing ``option_ticks[buffer.pos]`` lands in the slot add() is about to fill —
    aligning the discount with the transition's reward. No-ops if the buffer is not
    an SMDP buffer (e.g. recurrent_ppo) or the env lacks option_ticks (single-env).
    """

    def _find_env(self):
        env = self.training_env
        seen = 0
        while env is not None and not hasattr(env, "option_ticks") and seen < 8:
            env = getattr(env, "venv", None)
            seen += 1
        return env if (env is not None and hasattr(env, "option_ticks")) else None

    def _on_step(self) -> bool:
        buf = getattr(self.model, "rollout_buffer", None)
        if buf is None or not hasattr(buf, "option_ticks"):
            return True
        env = self._find_env()
        if env is None:
            return True
        pos = buf.pos
        if 0 <= pos < buf.buffer_size:
            buf.option_ticks[pos] = env.option_ticks()
        return True
