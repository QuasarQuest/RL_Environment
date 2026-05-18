# src/model/export.py
#
# Exports a trained SB3 PPO model to ONNX format for use in Rust (via `ort`).
#
# Usage:
#   python -m src.model.export --model runs/models/best_model.zip \
#                              --out   runs/models/policy.onnx
#
# The exported ONNX graph takes a float32 obs tensor [batch, 53] and outputs
# action logits [batch, 26]. In Rust, argmax over logits gives the action.
#
# Validation: runs a forward pass through both the SB3 model and the ONNX
# runtime and asserts outputs match within tolerance.

from __future__ import annotations

import numpy as np
import torch
import typer
from pathlib import Path
from stable_baselines3 import PPO

app = typer.Typer()


@app.command()
def export_onnx(
    model_path: Path = typer.Option(...,  "--model", help="Path to SB3 .zip checkpoint"),
    out_path:   Path = typer.Option(...,  "--out",   help="Output .onnx path"),
    opset:      int  = typer.Option(17,   "--opset", help="ONNX opset version"),
):
    """Export a trained PPO policy to ONNX for Rust inference."""
    import onnx
    import onnxruntime as ort

    print(f"Loading model from {model_path} ...")
    model = PPO.load(str(model_path))
    policy = model.policy
    policy.eval()

    obs_dim = policy.observation_space.shape[0]
    dummy   = torch.zeros(1, obs_dim)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Wrap policy to export only the action distribution forward pass ────────
    class PolicyExportWrapper(torch.nn.Module):
        def __init__(self, p):
            super().__init__()
            self.p = p

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            # Returns action logits (before softmax)
            features    = self.p.extract_features(obs, self.p.pi_features_extractor)
            latent_pi   = self.p.mlp_extractor.forward_actor(features)
            return self.p.action_net(latent_pi)

    wrapper = PolicyExportWrapper(policy)

    print(f"Exporting to {out_path} (opset {opset}) ...")
    torch.onnx.export(
        wrapper,
        dummy,
        str(out_path),
        opset_version     = opset,
        input_names       = ["obs"],
        output_names      = ["logits"],
        dynamic_axes      = {"obs": {0: "batch"}, "logits": {0: "batch"}},
        do_constant_folding = True,
    )

    # ── Validate ──────────────────────────────────────────────────────────────
    print("Validating ONNX model ...")
    onnx_model = onnx.load(str(out_path))
    onnx.checker.check_model(onnx_model)

    sess    = ort.InferenceSession(str(out_path))
    obs_np  = np.zeros((1, obs_dim), dtype=np.float32)
    onnx_out = sess.run(["logits"], {"obs": obs_np})[0]

    with torch.no_grad():
        torch_out = wrapper(torch.from_numpy(obs_np)).numpy()

    max_diff = float(np.abs(onnx_out - torch_out).max())
    assert max_diff < 1e-4, f"ONNX/PyTorch mismatch: {max_diff}"

    print(f"✓ Export validated (max diff = {max_diff:.2e})")
    print(f"✓ Saved to {out_path}")


if __name__ == "__main__":
    app()