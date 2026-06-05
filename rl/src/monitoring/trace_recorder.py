"""Offline behaviour trace recorder.

Runs a trained checkpoint on a single env and records, for every high-level
decision and every sim tick, exactly what the agent saw and chose — so reward
shaping (the balance between the dense-cluster actions and NavigateToNearestGold)
can be tuned against evidence instead of guesswork.

Why this exists
---------------
The agent over-uses `NavigateToNearestGold` (action 12). We want it to prefer
dense gold clusters unless gold is genuinely close. The intra-option SMDP discount
(`reward.option_gamma`) is meant to trade near-vs-far, but its effect on the
policy's choices is invisible. This tool makes it visible: at each decision it
logs the per-region gold counts + path distances (the obs cluster features), the
masked action probabilities, the value estimate, and the per-tick reward
breakdown (tick / pickup / deposit / wall) that the option earned.

What's faithful about the granularity
--------------------------------------
A navigation option runs many ticks via A* but ENDS the moment a pickup/deposit
fires (those change gold_carried/score — a stop condition in SimCore::step_option).
So within one option the gold/item map is essentially static: full positions are
snapshotted once per DECISION, while per TICK we record only what changes (agent
position + reward components). See sim/src/engine/mod.rs::TickRecord.

Output
------
A single HDF5 file (`<run>/trace.h5` by default). Layout — see `record` below.

CLI
---
    atb-trace --run runs/seq_s1_<id> --episodes 5          # record (default cmd)
    atb-trace summary runs/seq_s1_<id>/trace.h5            # human-readable summary

    # explicit paths instead of --run:
    atb-trace --model M/best_model.zip --vecnorm M/vn.pkl --config cfg.ron
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch as th
import typer

from env.atb_env import DEFAULT_CONFIG
from env.batch_vec_env import BatchVecEnv
from network.extractor import ACTION_SIZE, CLUSTER_FEATURES, CLUSTER_K

log = logging.getLogger(__name__)
app = typer.Typer(add_completion=False)

# Action names — must match sim/src/rl/action.rs (CLUSTER_K nav slots + 5 specials).
ACTION_NAMES: list[str] = (
    [f"NavCluster{k}" for k in range(CLUSTER_K)]
    + ["NavBase", "NavSpeed", "NavMultiplier", "NavNearestGold", "Wait"]
)
ACTION_NEAREST = CLUSTER_K + 3  # index of NavigateToNearestGold

# Per-tick trace column order — must match SimCore::trace_flat / TickRecord.
TICK_FIELDS: list[str] = [
    "tick", "ax", "ay", "gold_carried", "score",
    "r_tick", "r_pickup", "r_deposit", "r_wall", "r_total", "discount", "gold_count",
]

# Cluster-feature sub-layout: region k → obs offset k*4 + {dx, dy, pathdist, count}.
_CF_DX, _CF_DY, _CF_PATHDIST, _CF_COUNT = 0, 1, 2, 3


# ── Path resolution ──────────────────────────────────────────────────────────

def _resolve_paths(
    run: Optional[Path],
    model: Optional[Path],
    vecnorm: Optional[Path],
    out: Optional[Path],
) -> tuple[Path, Optional[Path], Path]:
    """Resolve (model_path, vecnorm_path|None, out_path) from --run or explicit flags.

    A run dir is laid out (see training/train.py): <run>/eval_best/best_model.zip
    and the matching <run>/eval_best_vecnorm.pkl sibling.
    """
    if model is None:
        if run is None:
            raise typer.BadParameter("Pass --run <dir> or --model <path>.")
        model = run / "eval_best" / "best_model.zip"
    if not model.exists():
        raise typer.BadParameter(f"Model not found: {model}")

    if vecnorm is None and run is not None:
        for cand in (run / "eval_best_vecnorm.pkl", run / "eval_best" / "best_model_vecnorm.pkl"):
            if cand.exists():
                vecnorm = cand
                break

    if out is None:
        base = run if run is not None else model.parent
        out = base / "trace.h5"
    return model, vecnorm, out


# ── Inference helper ──────────────────────────────────────────────────────────

def _policy_step(model, policy_obs: np.ndarray, mask: np.ndarray, deterministic: bool):
    """Return (probs[A], value, action) for one observation under the masked policy."""
    obs_t, _ = model.policy.obs_to_tensor(policy_obs)
    with th.no_grad():
        dist = model.policy.get_distribution(obs_t, action_masks=mask.reshape(1, -1))
        probs = dist.distribution.probs.detach().cpu().numpy()[0].astype(np.float32)
        value = float(model.policy.predict_values(obs_t).detach().cpu().numpy().reshape(-1)[0])
        if deterministic:
            action = int(probs.argmax())
        else:
            action = int(dist.get_actions(deterministic=False).detach().cpu().numpy().reshape(-1)[0])
    return probs, value, action


# ── Per-episode accumulator ───────────────────────────────────────────────────

class _Episode:
    """Buffers one episode's decision rows + per-tick rows, then writes a group."""

    def __init__(self, tiles: np.ndarray, grid_w: int, grid_h: int) -> None:
        self.map_tiles = tiles  # (H, W) uint8
        # Base tile code is 10 + team (see BatchEnv::get_tiles). Team 0 → 10.
        ys, xs = np.where((tiles >= 10) & (tiles < 20))
        self.base_x = int(xs[0]) if len(xs) else -1
        self.base_y = int(ys[0]) if len(ys) else -1
        self.dec: list[dict] = []
        self.gold_xy: list[np.ndarray] = []
        self.gold_off: list[int] = [0]
        self.item_xyk: list[np.ndarray] = []
        self.item_off: list[int] = [0]
        self.ticks: list[np.ndarray] = []      # each (n_tick, len(TICK_FIELDS))
        self.tick_dec_idx: list[np.ndarray] = []
        self.total_reward = 0.0

    def add_decision(self, row: dict, gold: np.ndarray, items: np.ndarray,
                     trace: np.ndarray) -> None:
        idx = len(self.dec)
        self.dec.append(row)
        self.total_reward += row["option_reward"]

        self.gold_xy.append(gold)
        self.gold_off.append(self.gold_off[-1] + len(gold))
        self.item_xyk.append(items)
        self.item_off.append(self.item_off[-1] + len(items))

        self.ticks.append(trace)
        self.tick_dec_idx.append(np.full(len(trace), idx, dtype=np.uint32))

    def write(self, f: h5py.File, ep_idx: int, final_score: float, length_ticks: int) -> None:
        g = f.create_group(f"episode_{ep_idx:03d}")
        g.attrs["final_score"] = float(final_score)
        g.attrs["length_ticks"] = int(length_ticks)
        g.attrs["n_decisions"] = len(self.dec)
        g.attrs["total_reward"] = float(self.total_reward)
        g.attrs["base_x"] = self.base_x
        g.attrs["base_y"] = self.base_y
        g.create_dataset("map_tiles", data=self.map_tiles, compression="gzip")

        dg = g.create_group("decisions")
        keys = self.dec[0].keys() if self.dec else []
        for k in keys:
            arr = np.array([d[k] for d in self.dec])
            dg.create_dataset(k, data=arr, compression="gzip")
        # Ragged gold/item positions, CSR-style (concat + offsets).
        dg.create_dataset("gold_xy", data=_vstack(self.gold_xy, 2, np.int16), compression="gzip")
        dg.create_dataset("gold_offsets", data=np.array(self.gold_off, dtype=np.int32))
        dg.create_dataset("item_xyk", data=_vstack(self.item_xyk, 3, np.int16), compression="gzip")
        dg.create_dataset("item_offsets", data=np.array(self.item_off, dtype=np.int32))

        tg = g.create_group("ticks")
        all_ticks = _vstack(self.ticks, len(TICK_FIELDS), np.float32)
        dec_idx = np.concatenate(self.tick_dec_idx) if self.tick_dec_idx else np.empty(0, np.uint32)
        tg.create_dataset("decision_idx", data=dec_idx, compression="gzip")
        for j, name in enumerate(TICK_FIELDS):
            tg.create_dataset(name, data=all_ticks[:, j], compression="gzip")


def _vstack(arrs: list[np.ndarray], ncol: int, dtype) -> np.ndarray:
    """Vertically stack rows, returning an empty (0, ncol) array when there are none."""
    nonempty = [a for a in arrs if len(a)]
    if not nonempty:
        return np.empty((0, ncol), dtype=dtype)
    return np.vstack(nonempty).astype(dtype)


# ── record command ─────────────────────────────────────────────────────────────

@app.callback(invoke_without_command=True)
def record(
    ctx: typer.Context,
    run: Optional[Path] = typer.Option(None, "--run", help="Run dir (derives model/vecnorm/out)."),
    model: Optional[Path] = typer.Option(None, "--model", help="Explicit model .zip."),
    vecnorm: Optional[Path] = typer.Option(None, "--vecnorm", help="Explicit VecNormalize .pkl."),
    config: Path = typer.Option(Path(DEFAULT_CONFIG), "--config", help="World .ron config."),
    out: Optional[Path] = typer.Option(None, "--out", help="Output .h5 (default <run>/trace.h5)."),
    episodes: int = typer.Option(5, "--episodes", help="Episodes to record."),
    algo: str = typer.Option("maskable_ppo", "--algo", help="ppo or maskable_ppo."),
    deterministic: bool = typer.Option(True, "--deterministic/--stochastic",
                                       help="argmax vs sample from the masked policy."),
) -> None:
    """Record per-decision + per-tick behaviour traces for a trained checkpoint.

    This is the default action; `summary` is the other subcommand.
    """
    # A named subcommand (e.g. `summary`) was invoked → don't also record.
    if ctx.invoked_subcommand is not None:
        return

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    import atb

    model_path, vn_path, out_path = _resolve_paths(run, model, vecnorm, out)
    log.info("model   : %s", model_path)
    log.info("vecnorm : %s", vn_path or "(none)")
    log.info("config  : %s", config)
    log.info("out     : %s", out_path)

    # ── Load policy (custom AtbCnnExtractor resolves via rl/src on path) ─────────
    if algo == "maskable_ppo":
        from sb3_contrib import MaskablePPO
        model_obj = MaskablePPO.load(str(model_path), device="cpu")
    else:
        from stable_baselines3 import PPO
        model_obj = PPO.load(str(model_path), device="cpu")

    # ── Single raw env + optional obs normalisation stats ────────────────────────
    raw = BatchVecEnv(1, str(config))
    raw._batch.set_trace(True)

    normalize_obs = None
    if vn_path is not None and vn_path.exists():
        from stable_baselines3.common.vec_env import VecNormalize
        vn = VecNormalize.load(str(vn_path), raw)
        vn.training = False
        # normalize_obs() is a no-op when the run had norm_obs=False, so this is
        # always correct: raw obs go to the policy unless obs-norm was trained.
        normalize_obs = vn.normalize_obs

    tf = atb.PyBatchEnv.trace_fields()
    assert tf == len(TICK_FIELDS), f"trace_fields {tf} != {len(TICK_FIELDS)}"
    grid_w, grid_h = raw._batch.grid_size()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        _write_meta(f, model_obj, model_path, config, episodes, deterministic, grid_w, grid_h, raw)

        obs = raw.reset()
        raw_obs = obs.copy()
        ep = _new_episode(raw, grid_w, grid_h)
        dec_idx = 0
        completed = 0

        while completed < episodes:
            agents = raw._batch.get_agents(0)
            ax, ay, _team, gold_carried, score = agents[0]
            items = np.array(raw._batch.get_items(0), dtype=np.int16).reshape(-1, 3)
            gold = items[items[:, 2] == 0][:, :2] if len(items) else np.empty((0, 2), np.int16)
            tick = raw._batch.get_tick(0)
            mask = np.asarray(raw._batch.action_masks(), dtype=bool).reshape(ACTION_SIZE)
            cf = raw_obs[0, -CLUSTER_FEATURES:].astype(np.float32)

            policy_obs = normalize_obs(raw_obs) if normalize_obs is not None else raw_obs
            probs, value, action = _policy_step(model_obj, policy_obs, mask, deterministic)

            raw.step_async(np.array([action], dtype=np.int64))
            obs, _rews, dones, infos = raw.step_wait()
            trace = np.asarray(raw._batch.get_trace(0), dtype=np.float32).reshape(-1, tf)
            dist_t, is_cluster_t, own_gold_t, skipped_t = raw.decision_telemetry()

            top5 = np.argsort(probs)[::-1][:5]
            cl = cf.reshape(CLUSTER_K, 4)
            row = {
                "tick": int(tick),
                "agent_x": int(ax), "agent_y": int(ay),
                "gold_carried": int(gold_carried), "score": int(score),
                "value": float(value),
                "chosen_action": int(action),
                "option_ticks": int(len(trace)),
                "option_reward": float(trace[:, 9].sum()) if len(trace) else 0.0,   # undiscounted Σ r_total
                "option_return": float((trace[:, 9] * trace[:, 10]).sum()) if len(trace) else 0.0,  # Σ γᵗ r
                "is_cluster": bool(is_cluster_t[0]),
                "own_has_gold": bool(own_gold_t[0]),
                "skipped_own": bool(skipped_t[0]),
                "chosen_dist": int(dist_t[0]),
                "action_probs": probs,
                "action_mask": mask,
                "top5_actions": top5.astype(np.uint8),
                "top5_probs": probs[top5].astype(np.float32),
                "cluster_dx": cl[:, _CF_DX].copy(),
                "cluster_dy": cl[:, _CF_DY].copy(),
                "cluster_pathdist": cl[:, _CF_PATHDIST].copy(),
                "cluster_count": cl[:, _CF_COUNT].copy(),
            }
            ep.add_decision(row, gold, items, trace)
            dec_idx += 1
            raw_obs = obs.copy()

            if bool(dones[0]):
                final_score = float(infos[0].get("score", score))
                length_ticks = int(trace[-1, 0]) if len(trace) else int(tick)
                ep.write(f, completed, final_score, length_ticks)
                log.info("episode %d: %d decisions, score=%.1f, reward=%.2f",
                         completed, len(ep.dec), final_score, ep.total_reward)
                completed += 1
                if completed < episodes:
                    ep = _new_episode(raw, grid_w, grid_h)
                    dec_idx = 0

    log.info("Wrote %s", out_path)


def _new_episode(raw: BatchVecEnv, grid_w: int, grid_h: int) -> _Episode:
    # PyO3 returns Vec<u8> as Python `bytes`, so decode via frombuffer.
    tiles = np.frombuffer(bytes(raw._batch.get_tiles(0)), dtype=np.uint8).reshape(grid_h, grid_w)
    return _Episode(tiles, grid_w, grid_h)


def _write_meta(f, model_obj, model_path, config, episodes, deterministic,
                grid_w, grid_h, raw) -> None:
    rt, rp, rd, rw, og = raw._batch.reward_weights()
    f.attrs["model_path"] = str(model_path)
    f.attrs["config_path"] = str(config)
    f.attrs["episodes"] = int(episodes)
    f.attrs["deterministic"] = bool(deterministic)
    f.attrs["grid_w"] = int(grid_w)
    f.attrs["grid_h"] = int(grid_h)
    f.attrs["cluster_k"] = int(CLUSTER_K)
    f.attrs["reward_tick"] = float(rt)
    f.attrs["reward_pickup"] = float(rp)
    f.attrs["reward_deposit"] = float(rd)
    f.attrs["reward_wall_hit"] = float(rw)
    f.attrs["option_gamma"] = float(og)
    f.attrs["action_names"] = np.array(ACTION_NAMES, dtype=h5py.string_dtype())
    f.attrs["tick_field_names"] = np.array(TICK_FIELDS, dtype=h5py.string_dtype())


# ── summary command ─────────────────────────────────────────────────────────────

@app.command()
def summary(
    path: Path = typer.Argument(..., help="A trace.h5 produced by `atb-trace record`."),
    dist_margin: float = typer.Option(0.10, "--dist-margin",
        help="Min extra normalised path-distance for a region to count as 'farther'."),
) -> None:
    """Print action frequencies and the near-vs-dense tradeoff the agent is making."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    chosen = []                  # chosen action per decision
    near_count, near_dist = [], []   # cluster_count / pathdist the agent committed to
    nearest_decisions = 0
    nearest_skipped_denser_far = 0   # NavNearestGold while a denser+farther region existed

    with h5py.File(path, "r") as f:
        names = [n.decode() if isinstance(n, bytes) else str(n) for n in f.attrs["action_names"]]
        eps = [k for k in f.keys() if k.startswith("episode_")]
        for ek in eps:
            d = f[ek]["decisions"]
            acts = d["chosen_action"][:]
            count = d["cluster_count"][:]      # (n, K)
            pdist = d["cluster_pathdist"][:]   # (n, K)
            chosen.append(acts)
            for i, a in enumerate(acts):
                a = int(a)
                if a < CLUSTER_K:                      # chose a cluster
                    near_count.append(float(count[i, a]))
                    near_dist.append(float(pdist[i, a]))
                if a == ACTION_NEAREST:
                    nearest_decisions += 1
                    # The nearest gold's region: smallest pathdist among regions with gold.
                    has = count[i] > 0
                    if has.any():
                        near_region = np.where(has, pdist[i], np.inf).argmin()
                        nd, nc = pdist[i, near_region], count[i, near_region]
                        denser_far = has & (count[i] > nc) & (pdist[i] > nd + dist_margin)
                        if denser_far.any():
                            nearest_skipped_denser_far += 1

    acts = np.concatenate(chosen) if chosen else np.empty(0, int)
    total = len(acts)
    log.info("\n=== %s — %d decisions over %d episodes ===", path.name, total, len(eps))
    log.info("\nAction frequency:")
    for a in range(len(names)):
        n = int((acts == a).sum())
        if n:
            log.info("  %-16s %6d  %5.1f%%", names[a], n, 100.0 * n / max(total, 1))

    if near_count:
        log.info("\nChosen-cluster decisions: mean count=%.3f  mean pathdist=%.3f",
                 float(np.mean(near_count)), float(np.mean(near_dist)))
    if nearest_decisions:
        frac = 100.0 * nearest_skipped_denser_far / nearest_decisions
        log.info("\nNavNearestGold decisions: %d", nearest_decisions)
        log.info("  ...where a DENSER region existed >%.2f farther: %d (%.1f%%)",
                 dist_margin, nearest_skipped_denser_far, frac)
        log.info("  (high %s ⇒ agent trades dense-far for sparse-near — raise pickup "
                 "weight or option_gamma to rebalance)", "%")


if __name__ == "__main__":
    app()
