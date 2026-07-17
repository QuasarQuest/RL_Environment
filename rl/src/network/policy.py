"""Policy kwargs for SB3 / sb3-contrib algorithm constructors.

Pass ATB_POLICY_KWARGS as ``policy_kwargs=`` when constructing PPO or
MaskablePPO. The extractor class is wired in here; callers never need to
import network.extractor directly.

Slim architecture notes
-----------------------
features_dim=256 is shared across all curriculum stages.
net_arch heads are intentionally lean — deep pi/vf heads add cost without benefit
for a 13-action discrete space.
"""
from __future__ import annotations

from network.extractor import AtbCnnExtractor

# normalize_images=False: obs are already float32 in [0,1]; SB3 must NOT ÷255.
ATB_POLICY_KWARGS = {
    "features_extractor_class": AtbCnnExtractor,
    "features_extractor_kwargs": {"features_dim": 256},
    "net_arch": {"pi": [64], "vf": [64]},  # slim heads — extractor carries the capacity
    "normalize_images": False,
}
