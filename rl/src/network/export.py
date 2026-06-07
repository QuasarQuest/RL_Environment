"""ONNX export for a trained policy.

Exports the deterministic actor path only: obs → (norm) → features → action_net.
Sampling, the value head, and exploration noise are Python-side concerns not
needed at deployment time.

When vecnorm_path is supplied and norm_obs=True, the VecNormalize obs running
stats are baked into the ONNX graph so the viewer can pass raw observations.

For RecurrentPPO (LSTM) policies the export is *stateful*:
  inputs:  obs [batch, obs_dim], h_in [n_layers, batch, hidden], c_in [n_layers, batch, hidden]
  outputs: logits [batch, n_actions], h_out [n_layers, batch, hidden], c_out [n_layers, batch, hidden]

A sidecar policy_meta.json is written alongside the ONNX file so the viewer
can reconstruct the correct h/c tensor shapes without inspecting the graph.

CLI
---
    atb-export --model runs/run_s1_XYZ/eval_best/best_model \\
               --out   runs/run_s1_XYZ/policy.onnx \\
               --vecnorm runs/run_s1_XYZ/eval_best_vecnorm.pkl
"""
from __future__ import annotations

import json
import logging
import pickle
import warnings
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
    """Stateless actor path — for non-LSTM policies."""

    def __init__(self, policy: torch.nn.Module) -> None:
        super().__init__()
        self.policy = policy

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.policy.extract_features(obs, self.policy.pi_features_extractor)  # type: ignore[arg-type]
        latent = self.policy.mlp_extractor.forward_actor(features)  # type: ignore[union-attr]
        return self.policy.action_net(latent)  # type: ignore[return-value]


class _NormPolicyWrapper(torch.nn.Module):
    """Stateless actor with baked-in VecNormalize obs stats — for non-LSTM policies."""

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
        obs = ((obs - self.obs_mean) / self.obs_std).clamp(-self.clip_obs, self.clip_obs)  # type: ignore[operator]
        features = self.policy.extract_features(obs, self.policy.pi_features_extractor)  # type: ignore[arg-type]
        latent = self.policy.mlp_extractor.forward_actor(features)  # type: ignore[union-attr]
        return self.policy.action_net(latent)  # type: ignore[return-value]


class _StatefulLstmWrapper(torch.nn.Module):
    """Stateful LSTM actor: (obs, h_in, c_in) → (logits, h_out, c_out).

    The viewer accumulates h/c across ticks so the policy retains temporal
    context for the full episode — matching how RecurrentPPO runs at training.
    Optionally bakes VecNormalize obs stats when norm_obs=True.
    """

    def __init__(
        self,
        policy: torch.nn.Module,
        mean: np.ndarray | None = None,
        var: np.ndarray | None = None,
        clip_obs: float = 10.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.policy = policy
        self.clip_obs = clip_obs
        self.norm = mean is not None
        if mean is not None and var is not None:
            std = np.sqrt(var + eps).astype(np.float32)
            self.register_buffer("obs_mean", torch.from_numpy(mean.astype(np.float32)))
            self.register_buffer("obs_std", torch.from_numpy(std))

    def forward(
        self,
        obs: torch.Tensor,
        h_in: torch.Tensor,
        c_in: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.norm:
            obs = ((obs - self.obs_mean) / self.obs_std).clamp(-self.clip_obs, self.clip_obs)  # type: ignore[operator]
        features = self.policy.extract_features(obs, self.policy.pi_features_extractor)  # type: ignore[arg-type]
        lstm: torch.nn.LSTM = self.policy.lstm_actor  # type: ignore[assignment]
        # features: [batch, feat] → [1, batch, feat] for seq_len=1
        out, (h_out, c_out) = lstm(features.unsqueeze(0), (h_in, c_in))
        features = out.squeeze(0)
        latent = self.policy.mlp_extractor.forward_actor(features)  # type: ignore[union-attr]
        logits = self.policy.action_net(latent)  # type: ignore[return-value]
        return logits, h_out, c_out


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

    has_lstm = hasattr(policy, "lstm_actor")

    # Resolve optional VecNormalize obs stats.
    mean_np: np.ndarray | None = None
    var_np: np.ndarray | None = None
    clip = 10.0
    if vecnorm_path is not None and vecnorm_path.exists():
        with open(vecnorm_path, "rb") as fh:
            vn = pickle.load(fh)
        if vn.norm_obs:
            mean_np = np.asarray(vn.obs_rms.mean, dtype=np.float32).reshape(obs_shape)
            var_np = np.asarray(vn.obs_rms.var, dtype=np.float32).reshape(obs_shape)
            clip = float(vn.clip_obs)
            log.info("VecNormalize obs stats baked into ONNX graph from %s", vecnorm_path)

    dummy = torch.randn(1, *obs_shape, dtype=torch.float32)

    with warnings.catch_warnings(record=False):
        warnings.filterwarnings("ignore", message=".*variable length with LSTM.*")

        if has_lstm:
            lstm: torch.nn.LSTM = policy.lstm_actor  # type: ignore[assignment]
            n_layers = lstm.num_layers
            hidden_size = lstm.hidden_size
            wrapper = _StatefulLstmWrapper(policy, mean_np, var_np, clip)
            dummy_h = torch.zeros(n_layers, 1, hidden_size)
            dummy_c = torch.zeros(n_layers, 1, hidden_size)
            # Static batch=1 (no dynamic_axes): the viewer runs single-obs inference,
            # and a symbolic batch dim leaves the feature-extractor Reshape (-1, C, H, W)
            # unresolvable for tract's startup analysis pass — it then falls back to the
            # BT agent instead of auto-loading the policy. Fixed shapes keep tract happy.
            torch.onnx.export(
                wrapper, (dummy, dummy_h, dummy_c), str(out_path),
                opset_version=opset,
                input_names=["obs", "h_in", "c_in"],
                output_names=["logits", "h_out", "c_out"],
                do_constant_folding=True,
                dynamo=False,
            )
            # Write sidecar so the viewer knows LSTM shape without parsing the graph.
            meta = {"lstm_n_layers": n_layers, "lstm_hidden_size": hidden_size}
            meta_path = out_path.with_name(out_path.stem + "_meta.json")
            meta_path.write_text(json.dumps(meta))
            log.info("LSTM metadata  → %s", meta_path)
        else:
            if mean_np is not None:
                assert var_np is not None  # mean_np and var_np are set together above
                wrapper = _NormPolicyWrapper(policy, mean_np, var_np, clip)
            else:
                wrapper = _PolicyWrapper(policy)
            # Static batch=1 (no dynamic_axes): the viewer runs single-obs inference,
            # and a symbolic batch dim leaves the feature-extractor Reshape (-1, C, H, W)
            # unresolvable for tract's startup analysis pass — it then falls back to the
            # BT agent instead of auto-loading the policy. Fixed shapes keep tract happy.
            torch.onnx.export(
                wrapper, (dummy,), str(out_path),
                opset_version=opset,
                input_names=["obs"],
                output_names=["logits"],
                do_constant_folding=True,
                dynamo=False,
            )

    onnx.checker.check_model(onnx.load(str(out_path)))

    # Validation — compare ONNX runtime vs torch with a single obs (seq_len=1 for LSTM).
    obs_np = np.random.RandomState(0).randn(1, *obs_shape).astype(np.float32)
    sess = ort.InferenceSession(str(out_path))
    if has_lstm:
        h_np = np.zeros([n_layers, 1, hidden_size], dtype=np.float32)
        c_np = np.zeros([n_layers, 1, hidden_size], dtype=np.float32)
        onnx_out = np.array(sess.run(["logits"], {"obs": obs_np, "h_in": h_np, "c_in": c_np})[0])
        with torch.no_grad():
            torch_out = wrapper(
                torch.from_numpy(obs_np),
                torch.from_numpy(h_np),
                torch.from_numpy(c_np),
            )[0].numpy()
    else:
        onnx_out = np.array(sess.run(["logits"], {"obs": obs_np})[0])
        with torch.no_grad():
            torch_out = wrapper(torch.from_numpy(obs_np)).numpy()

    max_diff = float(np.abs(onnx_out - torch_out).max())
    if max_diff >= tolerance:
        raise RuntimeError(
            f"ONNX/PyTorch mismatch above tolerance ({tolerance:.0e}): "
            f"max diff = {max_diff:.2e}"
        )

    log.debug("Validated (max diff = %.2e, tolerance = %.0e)", max_diff, tolerance)
    log.info("Saved → %s", out_path)


# ── CLI entry point ───────────────────────────────────────────────────────────

@app.command()
def export_onnx(
    model_path: Path = typer.Option(..., "--model"),
    out_path: Path = typer.Option(..., "--out"),
    # Match export_to_onnx's default (12) — the opset the eval/checkpoint callbacks
    # bake into the deployed policy.onnx that the viewer (tract) loads.
    opset: int = typer.Option(12, "--opset"),
    tolerance: float = typer.Option(1e-5, "--tolerance"),
    vecnorm_path: Optional[Path] = typer.Option(None, "--vecnorm"),
    algo: str = typer.Option("recurrent_ppo", "--algo", help="ppo, maskable_ppo, or recurrent_ppo"),
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
    elif algo == "recurrent_ppo":
        try:
            from sb3_contrib import RecurrentPPO
            policy = RecurrentPPO.load(str(model_path)).policy
        except ImportError:
            log.warning("sb3-contrib not installed — falling back to PPO loader")
            policy = PPO.load(str(model_path)).policy
    else:
        policy = PPO.load(str(model_path)).policy

    export_to_onnx(policy, out_path, opset, tolerance, vecnorm_path=vecnorm_path)


if __name__ == "__main__":
    app()
