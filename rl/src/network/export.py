"""ONNX export for a trained policy.

Exports the deterministic actor path only: obs → (norm) → features → action_net.
Sampling, the value head, and exploration noise are Python-side concerns not
needed at deployment time.

When vecnorm_path is supplied, the VecNormalize obs running stats are baked
into the ONNX graph so the viewer can pass raw observations directly.

CLI
---
    atb-export --model runs/run_s1_XYZ/eval_best/best_model \\
               --out   runs/run_s1_XYZ/policy.onnx \\
               --vecnorm runs/run_s1_XYZ/eval_best_vecnorm.pkl
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import typer
from stable_baselines3 import PPO

log = logging.getLogger(__name__)

app = typer.Typer(add_completion=False)


# ── Wrapper modules for ONNX tracing ─────────────────────────────────────────

class _PolicyWrapper(torch.nn.Module):
    """Actor path only — no value head, no sampling."""

    def __init__(self, policy: torch.nn.Module) -> None:
        super().__init__()
        self.policy = policy

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.policy.extract_features(obs, self.policy.pi_features_extractor)  # type: ignore[arg-type]
        latent = self.policy.mlp_extractor.forward_actor(features)  # type: ignore[union-attr]
        return self.policy.action_net(latent)  # type: ignore[return-value]


class _NormPolicyWrapper(torch.nn.Module):
    """Bakes VecNormalize obs stats into the graph, then runs the actor path.

    The viewer passes raw observations; this module applies the same
    (mean, std, clip) transform that VecNormalize used during training.
    """

    def __init__(
        self,
        policy: torch.nn.Module,
        mean: np.ndarray,
        var: np.ndarray,
        clip_obs: float = 10.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.policy = policy
        self.clip_obs = clip_obs
        std = np.sqrt(var + eps).astype(np.float32)
        self.register_buffer("obs_mean", torch.from_numpy(mean.astype(np.float32)))
        self.register_buffer("obs_std", torch.from_numpy(std))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = (obs - self.obs_mean) / self.obs_std  # type: ignore[operator]
        obs = obs.clamp(-self.clip_obs, self.clip_obs)
        features = self.policy.extract_features(obs, self.policy.pi_features_extractor)  # type: ignore[arg-type]
        latent = self.policy.mlp_extractor.forward_actor(features)  # type: ignore[union-attr]
        return self.policy.action_net(latent)  # type: ignore[return-value]


# ── Library function ──────────────────────────────────────────────────────────

def export_to_onnx(
    policy: torch.nn.Module,
    out_path: Path,
    opset: int = 12,
    tolerance: float = 1e-5,
    vecnorm_path: Path | None = None,
) -> None:
    """Export an SB3 ActorCriticPolicy to ONNX (actor path only) and validate."""
    import onnx
    import onnxruntime as ort

    policy.eval().cpu()
    obs_shape = policy.observation_space.shape  # type: ignore[union-attr]
    assert obs_shape is not None
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if vecnorm_path is not None and vecnorm_path.exists():
        with open(vecnorm_path, "rb") as fh:
            vn = pickle.load(fh)
        mean = np.asarray(vn.obs_rms.mean, dtype=np.float32).reshape(obs_shape)
        var = np.asarray(vn.obs_rms.var, dtype=np.float32).reshape(obs_shape)
        wrapper: torch.nn.Module = _NormPolicyWrapper(policy, mean, var, clip_obs=float(vn.clip_obs))
        log.info("VecNormalize stats baked into ONNX graph from %s", vecnorm_path)
    else:
        wrapper = _PolicyWrapper(policy)

    dummy = torch.randn(1, *obs_shape, dtype=torch.float32)
    # dynamo=False → legacy TorchScript exporter; does not require onnxscript.
    torch.onnx.export(
        wrapper, (dummy,), str(out_path),
        opset_version=opset,
        input_names=["obs"],
        output_names=["logits"],
        dynamic_axes={"obs": {0: "batch"}, "logits": {0: "batch"}},
        do_constant_folding=True,
        dynamo=False,
    )

    onnx.checker.check_model(onnx.load(str(out_path)))

    obs_np = np.random.RandomState(0).randn(4, *obs_shape).astype(np.float32)
    onnx_out = np.array(ort.InferenceSession(str(out_path)).run(["logits"], {"obs": obs_np})[0])
    with torch.no_grad():
        torch_out = wrapper(torch.from_numpy(obs_np)).numpy()

    max_diff = float(np.abs(onnx_out - torch_out).max())
    if max_diff >= tolerance:
        raise RuntimeError(
            f"ONNX/PyTorch mismatch above tolerance ({tolerance:.0e}): "
            f"max diff = {max_diff:.2e}"
        )

    log.info("Validated (max diff = %.2e, tolerance = %.0e)", max_diff, tolerance)
    log.info("Saved → %s", out_path)


# ── CLI entry point ───────────────────────────────────────────────────────────

@app.command()
def export_onnx(
    model_path: Path = typer.Option(..., "--model"),
    out_path: Path = typer.Option(..., "--out"),
    opset: int = typer.Option(17, "--opset"),
    tolerance: float = typer.Option(1e-5, "--tolerance"),
    vecnorm_path: Optional[Path] = typer.Option(None, "--vecnorm"),
    algo: str = typer.Option("maskable_ppo", "--algo", help="ppo or maskable_ppo"),
) -> None:
    """Export a saved PPO / MaskablePPO policy to ONNX and validate."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log.info("Loading %s  (algo=%s)", model_path, algo)

    if algo == "maskable_ppo":
        try:
            from sb3_contrib import MaskablePPO
            policy = MaskablePPO.load(str(model_path)).policy
        except ImportError:
            log.warning("sb3-contrib not installed — falling back to PPO loader")
            policy = PPO.load(str(model_path)).policy
    else:
        policy = PPO.load(str(model_path)).policy

    export_to_onnx(policy, out_path, opset, tolerance, vecnorm_path=vecnorm_path)


if __name__ == "__main__":
    app()
