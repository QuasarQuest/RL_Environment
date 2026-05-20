# rl/src/test_sim.py
#
# Smoke-test for the full Rust → Python (atb / PyRlEnv) pipeline.
#
# Verifies:
#   1. Static metadata (obs_shape, obs_dim, action_size) are self-consistent.
#   2. The env constructs successfully with an auto-located config.ron.
#   3. reset() returns a flat vector that reshapes to (C, H, W) without data loss.
#   4. Per-channel invariants hold at every reset:
#        - All values in [0, 1]
#        - OOB / own_base / gold channels are non-negative
#        - Channel SELF is exactly 1.0 at the crop centre, 0 elsewhere
#        - Channel CARRYING is all zeros (no gold held at episode start)
#   5. step() returns (obs, reward, done) with matching obs length.
#   6. A full episode using the Wait action runs to completion and reports
#      a sensible total reward (only tick-penalty fired).
#   7. reset() after a completed episode produces a valid initial observation.
#   8. Ten random actions exercise the full action space without raising.
#
# Run with:
#   python -m test_sim          # from rl/src/
#   python rl/src/test_sim.py   # from the project root

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Iterator

import numpy as np

import atb

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EnvConfig:
    """All constants that describe the environment contract in one place."""

    # Action index for the no-op / Wait action.
    wait_action: int = 25

    # Reward magnitude per tick for the tick-penalty channel.
    tick_penalty: float = -0.001

    # Tolerance when comparing total reward to the expected tick-penalty floor.
    reward_tolerance: float = 0.01

    # Number of random-action steps in the smoke test.
    random_steps: int = 10

    # How often (in ticks) to log progress during a full episode.
    log_every_n_ticks: int = 100

    # Human-readable names for the observation channels, in order.
    channel_names: tuple[str, ...] = field(
        default=("oob", "own_base", "gold", "self", "carrying")
    )


class ObsChannel(IntEnum):
    """Indices into the C dimension of the (C, H, W) observation tensor."""
    OOB      = 0
    OWN_BASE = 1
    GOLD     = 2
    SELF     = 3
    CARRYING = 4


# ---------------------------------------------------------------------------
# Config-file discovery
# ---------------------------------------------------------------------------

def find_config() -> Path:
    """
    Walk upward from this file to locate ``assets/world/config.ron``.

    Returns the first match found, or raises ``FileNotFoundError`` with a
    descriptive message listing the search root.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "assets" / "world" / "config.ron"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Cannot find assets/world/config.ron in any ancestor of {here}"
    )


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------

def reshape_obs(flat: list[float], shape: tuple[int, int, int]) -> np.ndarray:
    """Convert a flat observation list to a (C, H, W) float32 array."""
    return np.asarray(flat, dtype=np.float32).reshape(shape)


def assert_obs_invariants(
    obs: np.ndarray,
    cfg: EnvConfig,
    *,
    label: str = "obs",
) -> None:
    """
    Assert the per-channel invariants that must hold at every episode start.

    Parameters
    ----------
    obs:
        A (C, H, W) float32 observation array.
    cfg:
        Environment configuration (provides channel names for error messages).
    label:
        Human-readable identifier used in assertion messages.
    """
    c, h, w = obs.shape
    centre = h // 2

    # Global range check.
    obs_min, obs_max = float(obs.min()), float(obs.max())
    assert 0.0 <= obs_min and obs_max <= 1.0, (
        f"[{label}] Values out of [0, 1]: min={obs_min}, max={obs_max}"
    )

    # Channel SELF: single hot pixel at the crop centre.
    self_ch = obs[ObsChannel.SELF]
    assert self_ch[centre, centre] == 1.0, (
        f"[{label}] SELF channel centre = {self_ch[centre, centre]:.4f}, expected 1.0"
    )
    assert self_ch.sum() == 1.0, (
        f"[{label}] SELF channel sum = {self_ch.sum():.4f}, expected exactly 1.0"
    )

    # Channel CARRYING: no gold held at the start of an episode.
    carry_sum = float(obs[ObsChannel.CARRYING].sum())
    assert carry_sum == 0.0, (
        f"[{label}] CARRYING channel non-zero at reset: sum={carry_sum}"
    )

    # Log per-channel sums for visual sanity.
    log.info("[%s] channel sums:", label)
    for i, name in enumerate(cfg.channel_names):
        log.info("  ch%d %-9s = %.2f", i, name, obs[i].sum())


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    env: atb.PyRlEnv,  # type: ignore[name-defined]
    action: int,
    obs_dim: int,
) -> Iterator[tuple[int, float, bool]]:
    """
    Run a single episode using a fixed action, yielding (tick, reward, done).

    The caller is responsible for calling ``env.reset()`` beforehand.
    Iteration stops automatically when ``done`` is ``True``.
    """
    done = False
    tick = 0
    while not done:
        obs_flat, reward, done = env.step(action)
        assert len(obs_flat) == obs_dim, (
            f"step() returned obs of length {len(obs_flat)}, expected {obs_dim}"
        )
        tick += 1
        yield tick, float(reward), bool(done)


# ---------------------------------------------------------------------------
# Sub-tests
# ---------------------------------------------------------------------------

def test_static_metadata() -> tuple[tuple[int, int, int], int, int]:
    """
    Verify self-consistency of the static class-level metadata.

    Returns ``(shape, obs_dim, action_size)`` for downstream use.
    """
    shape: tuple[int, int, int] = atb.PyRlEnv.obs_shape()  # type: ignore[attr-defined]
    obs_dim: int                = atb.PyRlEnv.obs_dim()     # type: ignore[attr-defined]
    action_size: int            = atb.PyRlEnv.action_size() # type: ignore[attr-defined]

    log.info("obs_shape   = %s", shape)
    log.info("obs_dim     = %d  (C*H*W = %d)", obs_dim, shape[0] * shape[1] * shape[2])
    log.info("action_size = %d", action_size)

    assert obs_dim == shape[0] * shape[1] * shape[2], (
        f"obs_dim {obs_dim} does not match shape product {shape}"
    )
    c, h, w = shape
    assert h == w, f"Crop should be square, got {h}×{w}"

    return shape, obs_dim, action_size


def test_reset(
    env: atb.PyRlEnv,  # type: ignore[name-defined]
    shape: tuple[int, int, int],
    obs_dim: int,
    cfg: EnvConfig,
    *,
    label: str = "reset",
) -> np.ndarray:
    """
    Call reset(), reshape the observation, check invariants, and return the array.
    """
    obs_flat = env.reset()
    assert len(obs_flat) == obs_dim, (
        f"[{label}] reset() returned obs of length {len(obs_flat)}, expected {obs_dim}"
    )
    obs = reshape_obs(obs_flat, shape)
    log.info(
        "[%s] shape=%s  min=%.3f  max=%.3f",
        label, obs.shape, obs.min(), obs.max(),
    )
    assert_obs_invariants(obs, cfg, label=label)
    return obs


def test_full_episode(
    env: atb.PyRlEnv,  # type: ignore[name-defined]
    shape: tuple[int, int, int],
    obs_dim: int,
    cfg: EnvConfig,
) -> None:
    """Run a complete Wait-action episode and verify the total reward is sane."""
    log.info("--- Full episode (Wait action=%d) ---", cfg.wait_action)

    test_reset(env, shape, obs_dim, cfg, label="pre-episode reset")

    total_reward = 0.0
    final_tick = 0

    for tick, reward, done in run_episode(env, cfg.wait_action, obs_dim):
        total_reward += reward
        final_tick = tick
        if tick % cfg.log_every_n_ticks == 0 or done:
            log.info(
                "  tick=%4d  reward=%+.4f  total=%+.3f  done=%s",
                tick, reward, total_reward, done,
            )

    log.info(
        "Episode finished — ticks=%d  total_reward=%+.3f",
        final_tick, total_reward,
    )

    expected_floor = cfg.tick_penalty * final_tick
    deviation = abs(total_reward - expected_floor)
    if deviation > cfg.reward_tolerance:
        log.warning(
            "Total reward %+.3f deviates from pure tick-penalty floor %+.3f "
            "(diff=%.4f). If the agent stumbled onto gold, that is expected.",
            total_reward, expected_floor, deviation,
        )


def test_reset_after_episode(
    env: atb.PyRlEnv,  # type: ignore[name-defined]
    shape: tuple[int, int, int],
    obs_dim: int,
    cfg: EnvConfig,
) -> None:
    """Verify that reset() works correctly after a completed episode."""
    log.info("--- Reset after episode ---")
    test_reset(env, shape, obs_dim, cfg, label="post-episode reset")


def test_random_actions(
    env: atb.PyRlEnv,  # type: ignore[name-defined]
    obs_dim: int,
    action_size: int,
    cfg: EnvConfig,
) -> None:
    """
    Step through a fixed number of random actions.

    The episode is restarted automatically if ``done`` is signalled mid-run.
    """
    log.info("--- Random-action smoke (%d steps) ---", cfg.random_steps)

    rng = np.random.default_rng(seed=0)
    rewards: list[float] = []

    for _ in range(cfg.random_steps):
        action = int(rng.integers(0, action_size))
        obs_flat, reward, done = env.step(action)
        assert len(obs_flat) == obs_dim, (
            f"step() returned obs of length {len(obs_flat)}, expected {obs_dim}"
        )
        rewards.append(float(reward))
        if done:
            env.reset()

    log.info("rewards: %s", [round(r, 4) for r in rewards])


def test_gold_reachability(
    env: atb.PyRlEnv,  # type: ignore[name-defined]
    obs_dim: int,
    action_size: int,
    *,
    max_steps: int = 2000,
) -> None:
    """
    Assert that a positive reward is reachable under random exploration.

    This is a prerequisite check for RL training: if no random policy can
    ever collect gold within ``max_steps``, the reward signal is broken and
    training will never converge.

    The test fails hard if no positive reward is observed; it logs the
    collection step and the reward value when one is found.
    """
    log.info("--- Gold reachability (max %d steps) ---", max_steps)

    env.reset()
    rng = np.random.default_rng(seed=42)
    episodes = 0

    for step_n in range(1, max_steps + 1):
        action = int(rng.integers(0, action_size))
        obs_flat, reward, done = env.step(action)
        assert len(obs_flat) == obs_dim, (
            f"step() returned obs of length {len(obs_flat)}, expected {obs_dim}"
        )
        if reward > 0.0:
            log.info(
                "  Gold collected at step %d (episode %d) — reward=%+.4f ✓",
                step_n, episodes, reward,
            )
            return
        if done:
            episodes += 1
            env.reset()

    # No positive reward found — this is a hard failure.
    raise AssertionError(
        f"No positive reward observed in {max_steps} steps across {episodes} episodes. "
        "The reward signal may be broken: check gold spawn logic, pickup radius, "
        "and the CARRYING / delivery reward in the Rust simulation."
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = EnvConfig()

    # 1. Static metadata
    shape, obs_dim, action_size = test_static_metadata()

    # 2. Construct env
    config_path = find_config()
    log.info("config_path = %s", config_path)
    env = atb.PyRlEnv(str(config_path))  # type: ignore[attr-defined]

    # 3. Full episode (includes its own pre-episode reset)
    test_full_episode(env, shape, obs_dim, cfg)

    # 4. Reset after episode
    test_reset_after_episode(env, shape, obs_dim, cfg)

    # 5. Random-action smoke
    test_random_actions(env, obs_dim, action_size, cfg)

    # 6. Gold reachability — hard-fails if positive reward is never observed
    test_gold_reachability(env, obs_dim, action_size)

    log.info("All checks passed.")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, FileNotFoundError, RuntimeError) as exc:
        log.error("Test failed: %s", exc)
        sys.exit(1)