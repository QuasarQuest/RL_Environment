"""Bootstrap a 2-frame-stack PPO model from a single-frame checkpoint.

The first conv layer grows from 9 → 18 input channels (one extra frame).
Old weights fill the first 9 channels; the second 9 are zero-initialised so
the model starts behaving exactly like the old single-frame policy and
gradually learns to exploit temporal context during training.

All other layers (conv 2/3, Linear head, pi/vf nets, action_net) are copied
exactly — their shapes are unchanged.

Usage
-----
    cd rl
    PYTHONPATH=src python -m model.bootstrap \\
        --model  ../runs/models/run_s1_XYZ_eval_best/best_model \\
        --vecnorm ../runs/models/run_s1_XYZ_eval_best_vecnorm.pkl \\
        --out    ../runs/models/bootstrap_s1_start
"""
from __future__ import annotations

import pickle
from pathlib import Path

import typer
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize

from env.atb_env import AtbEnv, DEFAULT_CONFIG
from model.policy import ATB_POLICY_KWARGS

app = typer.Typer(add_completion=False)

FRAME_STACK = 2

def _is_first_conv(key: str) -> bool:
    """True for any extractor's first conv weight (pi/vf/shared all need channel-expand)."""
    return key.endswith("conv.0.weight") and "features_extractor" in key


def _find_vecnormalize(env) -> VecNormalize | None:
    while env is not None:
        if isinstance(env, VecNormalize):
            return env
        env = getattr(env, "venv", None)
    return None


def _build_env(config_path: str = DEFAULT_CONFIG) -> VecFrameStack:
    vec = DummyVecEnv([lambda: AtbEnv(config_path)])
    vec = VecNormalize(vec, norm_obs=True, norm_reward=False, training=False)
    vec = VecFrameStack(vec, n_stack=FRAME_STACK, channels_order="first")
    return vec


@app.command()
def bootstrap(
    model_path:   Path = typer.Option(..., "--model",   help="Old single-frame model (.zip or stem)"),
    vecnorm_path: Path = typer.Option(..., "--vecnorm", help="Matching _vecnorm.pkl from old run"),
    out:          Path = typer.Option(..., "--out",     help="Output stem (no extension)"),
) -> None:
    """Transfer weights from a 9-channel model into a new 18-channel (2-frame) model."""

    # ── Load old model ────────────────────────────────────────────────────────
    print(f"Loading old model: {model_path}")
    old = PPO.load(str(model_path))
    old_sd = old.policy.state_dict()

    # ── Build new env (18-channel obs via frame stack) ────────────────────────
    print("Building env (n_frame_stack=2) …")
    env = _build_env()
    print(f"  obs shape: {env.observation_space.shape}")

    # ── Seed new model with correct architecture ──────────────────────────────
    print("Creating new PPO model …")
    new_model = PPO(
        policy="CnnPolicy",
        env=env,
        policy_kwargs=ATB_POLICY_KWARGS,
        verbose=0,
    )
    new_sd = new_model.policy.state_dict()

    # ── Transfer weights ──────────────────────────────────────────────────────
    transferred, zero_init, skipped = [], [], []

    for key in new_sd:
        if _is_first_conv(key) and key in old_sd and old_sd[key].shape != new_sd[key].shape:
            old_w = old_sd[key]           # (32, 9, 3, 3)
            new_w = new_sd[key].clone()   # (32, 18, 3, 3)
            c_old = old_w.shape[1]
            new_w[:, :c_old, :, :] = old_w
            new_w[:, c_old:, :, :] = 0.0
            new_sd[key] = new_w
            transferred.append(f"{key}  {list(old_w.shape)} → {list(new_w.shape)}")
            zero_init.append(f"  second {new_w.shape[1] - c_old} channels zero-initialised")
        elif key in old_sd and old_sd[key].shape == new_sd[key].shape:
            new_sd[key] = old_sd[key]
            transferred.append(key)
        else:
            skipped.append(key)

    new_model.policy.load_state_dict(new_sd)

    print(f"\nTransferred {len(transferred)} tensors:")
    for name in transferred:
        print(f"  ✓ {name}")
    for note in zero_init:
        print(note)
    if skipped:
        print(f"\nSkipped (no shape match in old model):")
        for name in skipped:
            print(f"  - {name}")

    # ── Copy old VecNorm obs stats into the new env ───────────────────────────
    vn = _find_vecnormalize(env)
    if vn is not None and vecnorm_path.exists():
        with open(vecnorm_path, "rb") as fh:
            old_vn = pickle.load(fh)
        vn.obs_rms = old_vn.obs_rms
        vn.ret_rms = old_vn.ret_rms
        print(f"\nCopied vecnorm stats ← {vecnorm_path}")
    elif not vecnorm_path.exists():
        print(f"\n[warn] vecnorm not found at {vecnorm_path} — new model starts with fresh stats")

    # ── Save ──────────────────────────────────────────────────────────────────
    out.parent.mkdir(parents=True, exist_ok=True)
    new_model.save(str(out))
    print(f"\n✓ Bootstrap model → {out}.zip")

    if vn is not None:
        vn_out = Path(str(out) + "_vecnorm.pkl")
        vn.save(str(vn_out))
        print(f"✓ VecNorm stats  → {vn_out}")

    env.close()

    print(
        f"\nResume with:\n"
        f"  atb-train +train.resume={out}"
    )


if __name__ == "__main__":
    app()
