"""Policy kwargs for SB3 / sb3-contrib algorithm constructors.

Pass ATB_POLICY_KWARGS as ``policy_kwargs=`` when constructing PPO or
MaskablePPO. The extractor class is wired in here; callers never need to
import network.extractor directly.

To switch back to the flat-MLP pipeline (useful for ablations):
    from network.policy import ATB_MLP_POLICY_KWARGS
    model = PPO(..., policy_kwargs=ATB_MLP_POLICY_KWARGS)

Slim architecture notes
-----------------------
features_dim=256 is shared across all 6 curriculum stages.
The extractor itself is now ~1.19M params (down from ~12.65M).
LSTM hidden size is controlled via ppo config YAML (lstm_hidden_size: 128).
net_arch heads are intentionally lean — the LSTM already provides a rich
temporal representation; deep pi/vf heads add cost without benefit here.
"""
from __future__ import annotations

from stable_baselines3.common.policies import ActorCriticPolicy

from network.extractor import AtbCnnExtractor, AtbMlpExtractor

# ── MLP (legacy) ──────────────────────────────────────────────────────────────

ATB_MLP_POLICY_KWARGS = dict(
    features_extractor_class=AtbMlpExtractor,
    features_extractor_kwargs=dict(features_dim=128),
    net_arch=dict(pi=[64], vf=[64]),
)

# ── CNN + minimap (current, non-recurrent) ────────────────────────────────────
# normalize_images=False: obs are already float32 in [0,1]; SB3 must NOT ÷255.

ATB_CNN_POLICY_KWARGS = dict(
    features_extractor_class=AtbCnnExtractor,
    features_extractor_kwargs=dict(features_dim=256),
    net_arch=dict(pi=[64], vf=[64]),  # slim: was [128, 64] — LSTM replaces depth
    normalize_images=False,
)

# ── RecurrentPPO (LSTM) ───────────────────────────────────────────────────────
# Uses MlpLstmPolicy — AtbCnnExtractor feeds into the LSTM layer.
# normalize_images=False: obs are already float32 in [0,1]; SB3 must NOT ÷255.
# lstm_hidden_size / n_lstm_layers are injected by train.py from PpoConfig,
# so they are NOT set here — keeping this dict minimal avoids surprises when
# overriding from YAML.

ATB_RECURRENT_POLICY_KWARGS = dict(
    features_extractor_class=AtbCnnExtractor,
    features_extractor_kwargs=dict(features_dim=256),
    normalize_images=False,
)

# Canonical alias — swap RHS to switch pipelines.
ATB_POLICY_KWARGS = ATB_CNN_POLICY_KWARGS

# Re-exported for callers that want the policy class directly.
AtbPolicy = ActorCriticPolicy
