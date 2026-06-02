"""Policy-behaviour telemetry → TensorBoard.

Answers the question "does the agent skip nearby gold and range far?" by logging,
per rollout:

  policy/chosen_gold_dist         mean Chebyshev distance to the gold the policy
                                  committed to collect (high = ranging far)
  policy/own_region_skip_rate     when the agent's own region held collectible gold
                                  and it chose to collect gold, the fraction of the
                                  time it picked a DIFFERENT region instead
  policy/own_region_has_gold_rate fraction of decisions where the agent's own region
                                  had collectible gold (context for the skip rate)
  policy/option_len               mean option length (sim ticks per high-level
                                  decision) — the direct readout of whether the SMDP
                                  discount + nearest-gold action are shortening trips;
                                  also flags wandering if it pins near MAX_OPTION_TICKS

Telemetry is computed in Rust at each option boundary (SimCore::decision_telem) and
surfaced through BatchVecEnv.decision_telemetry(). Only the batched training env
exposes it; the single-env DummyVecEnv path is silently skipped.
"""
from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback


class PolicyTelemetryCallback(BaseCallback):
    """Accumulate decision telemetry each step; log per-rollout means to TB."""

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._reset_accum()

    def _reset_accum(self) -> None:
        self._dist_sum = 0.0
        self._dist_n = 0
        self._skip_num = 0
        self._skip_den = 0
        self._own_gold_n = 0
        self._dec_n = 0
        self._optlen_sum = 0.0
        self._optlen_n = 0

    def _find_telemetry_env(self):
        """Unwrap VecNormalize etc. to the env exposing decision_telemetry()."""
        env = self.training_env
        seen = 0
        while env is not None and not hasattr(env, "decision_telemetry") and seen < 8:
            env = getattr(env, "venv", None)
            seen += 1
        return env if (env is not None and hasattr(env, "decision_telemetry")) else None

    def _on_step(self) -> bool:
        env = self._find_telemetry_env()
        if env is None:
            return True

        dist, is_cluster, own_gold, skipped = env.decision_telemetry()

        # Mean distance over gold-collection decisions with a valid target.
        gold_dec = is_cluster & (dist >= 0)
        self._dist_sum += float(dist[gold_dec].sum())
        self._dist_n += int(gold_dec.sum())

        # Skip rate: among decisions where local gold existed AND the agent chose to
        # collect gold, how often it went to a different region.
        den = own_gold & is_cluster
        self._skip_num += int((den & skipped).sum())
        self._skip_den += int(den.sum())

        self._own_gold_n += int(own_gold.sum())
        self._dec_n += int(dist.shape[0])

        # Mean option length (sim ticks per decision) — same batched env.
        if hasattr(env, "option_ticks"):
            ot = env.option_ticks()
            self._optlen_sum += float(ot.sum())
            self._optlen_n += int(ot.size)
        return True

    def _on_rollout_end(self) -> None:
        if self._dist_n:
            self.logger.record("policy/chosen_gold_dist", self._dist_sum / self._dist_n)
        if self._skip_den:
            self.logger.record("policy/own_region_skip_rate", self._skip_num / self._skip_den)
        if self._dec_n:
            self.logger.record("policy/own_region_has_gold_rate", self._own_gold_n / self._dec_n)
        if self._optlen_n:
            self.logger.record("policy/option_len", self._optlen_sum / self._optlen_n)
        self._reset_accum()
