"""ONNX export for the trained policy.

Exports just the deterministic action-logits path: `obs → features →
action_net`. Sampling, value head, and exploration noise stay on the
Python side because they are not needed at deployment time.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import typer
from stable_baselines3 import PPO

app = typer.Typer()


class _PolicyWrapper(torch.nn.Module):
    """Wraps `ActorCriticPolicy` for ONNX export of the actor path only."""

    def __init__(self, policy: torch.nn.Module) -> None:
        super().__init__()
        self.policy = policy

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.policy.extract_features(obs, self.policy.pi_features_extractor)
        latent = self.policy.mlp_extractor.forward_actor(features)
        return self.policy.action_net(latent)


@app.command()
def export_onnx(
    model_path: Path = typer.Option(..., "--model"),
    out_path: Path = typer.Option(..., "--out"),
    opset: int = typer.Option(17, "--opset"),
    tolerance: float = typer.Option(1e-5, "--tolerance"),
) -> None:
    """Export a saved PPO policy to ONNX and validate against PyTorch."""
    import onnx
    import onnxruntime as ort

    print(f"Loading {model_path}")
    policy = PPO.load(str(model_path)).policy
    policy.eval()  # critical: LayerNorm/BatchNorm in inference mode

    obs_dim = policy.observation_space.shape[0]
    wrapper = _PolicyWrapper(policy)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Use a non-zero dummy input so we exercise the LayerNorm code path
    # rather than the degenerate all-zeros case.
    dummy = torch.randn(1, obs_dim, dtype=torch.float32)

    torch.onnx.export(
        wrapper,
        dummy,
        str(out_path),
        opset_version=opset,
        input_names=["obs"],
        output_names=["logits"],
        dynamic_axes={"obs": {0: "batch"}, "logits": {0: "batch"}},
        do_constant_folding=True,
    )

    onnx.checker.check_model(onnx.load(str(out_path)))

    # Cross-check against PyTorch on a fresh random batch.
    obs_np = np.random.RandomState(0).randn(4, obs_dim).astype(np.float32)
    onnx_out = ort.InferenceSession(str(out_path)).run(["logits"], {"obs": obs_np})[0]
    with torch.no_grad():
        torch_out = wrapper(torch.from_numpy(obs_np)).numpy()

    max_diff = float(np.abs(onnx_out - torch_out).max())
    if max_diff >= tolerance:
        raise RuntimeError(
            f"ONNX/PyTorch mismatch above tolerance ({tolerance:.0e}): "
            f"max diff = {max_diff:.2e}"
        )

    print(f"✓ Validated (max diff = {max_diff:.2e}, tolerance = {tolerance:.0e})")
    print(f"✓ Saved to {out_path}")


if __name__ == "__main__":
    app()
