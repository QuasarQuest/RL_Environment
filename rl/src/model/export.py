from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import typer
from stable_baselines3 import PPO

app = typer.Typer()


class _PolicyWrapper(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.p = policy

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.p.extract_features(obs, self.p.pi_features_extractor)
        latent = self.p.mlp_extractor.forward_actor(features)
        return self.p.action_net(latent)


@app.command()
def export_onnx(
    model_path: Path = typer.Option(..., "--model"),
    out_path: Path = typer.Option(..., "--out"),
    opset: int = typer.Option(17, "--opset"),
) -> None:
    import onnx
    import onnxruntime as ort

    print(f"Loading {model_path}")
    policy = PPO.load(str(model_path)).policy
    policy.eval()

    obs_dim = policy.observation_space.shape[0]
    wrapper = _PolicyWrapper(policy)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        torch.zeros(1, obs_dim),
        str(out_path),
        opset_version=opset,
        input_names=["obs"],
        output_names=["logits"],
        dynamic_axes={"obs": {0: "batch"}, "logits": {0: "batch"}},
        do_constant_folding=True,
    )

    onnx.checker.check_model(onnx.load(str(out_path)))

    obs_np = np.zeros((1, obs_dim), dtype=np.float32)
    onnx_out = ort.InferenceSession(str(out_path)).run(["logits"], {"obs": obs_np})[0]
    with torch.no_grad():
        torch_out = wrapper(torch.from_numpy(obs_np)).numpy()

    max_diff = float(np.abs(onnx_out - torch_out).max())
    if max_diff >= 1e-4:
        raise RuntimeError(f"ONNX/PyTorch mismatch: {max_diff:.2e}")

    print(f"✓ Validated (max diff={max_diff:.2e})")
    print(f"✓ Saved to {out_path}")


if __name__ == "__main__":
    app()