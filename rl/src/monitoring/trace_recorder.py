"""Offline behaviour trace recorder.

Runs a trained checkpoint on a single env and records, for every high-level
decision and every sim tick, exactly what the agent saw and chose — so reward
shaping and the agent's spatial (which-region) decisions can be tuned against
evidence instead of guesswork.

Why this exists
---------------
The agent must reach gold through the fixed-region nav slots (there is no
hardcoded "nearest gold" action — see sim/src/rl/action.rs), so where it chooses
to go is a learned decision over the per-region obs features. We want to see
whether it steers toward dense clusters or just the nearest sparse gold. The
intra-option SMDP discount (`reward.option_gamma`) is meant to trade near-vs-far,
but its effect on the policy's choices is invisible. This tool makes it visible:
at each decision it logs the per-region gold counts + path distances (the obs
cluster features), the masked action probabilities, the value estimate, and the
per-tick reward breakdown (tick / pickup / deposit / wall) that the option earned.

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
    atb-trace view    runs/seq_s1_<id>/trace.h5 -e 0       # visual per-decision panels

    # explicit paths instead of --run:
    atb-trace --model M/best_model.zip --vecnorm M/vn.pkl --config cfg.ron
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, cast

import h5py
import numpy as np
import torch as th
import typer

from env.atb_env import DEFAULT_CONFIG
from env.batch_vec_env import BatchVecEnv
from network.extractor import ACTION_SIZE, CLUSTER_FEATURES, CLUSTER_K

log = logging.getLogger(__name__)
app = typer.Typer(add_completion=False)

# Action names — must match sim/src/rl/action.rs (CLUSTER_K nav slots + 4 specials).
ACTION_NAMES: list[str] = (
    [f"NavCluster{k}" for k in range(CLUSTER_K)]
    + ["NavBase", "NavSpeed", "NavMultiplier", "Wait"]
)

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

    def __init__(self, tiles: np.ndarray) -> None:
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

    This is the default action; `summary` and `view` are the other subcommands.
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
    raw.batch.set_trace(True)

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
    grid_w, grid_h = raw.batch.grid_size()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        _write_meta(f, model_path, config, episodes, deterministic, grid_w, grid_h, raw)

        obs = raw.reset()
        raw_obs = obs.copy()
        ep = _new_episode(raw, grid_w, grid_h)
        dec_idx = 0
        completed = 0

        while completed < episodes:
            agents = raw.batch.get_agents(0)
            ax, ay, _team, gold_carried, score = agents[0]
            items = np.array(raw.batch.get_items(0), dtype=np.int16).reshape(-1, 3)
            gold = items[items[:, 2] == 0][:, :2] if len(items) else np.empty((0, 2), np.int16)
            tick = raw.batch.get_tick(0)
            mask = np.asarray(raw.batch.action_masks(), dtype=bool).reshape(ACTION_SIZE)
            cf = raw_obs[0, -CLUSTER_FEATURES:].astype(np.float32)

            policy_obs = (
                cast(np.ndarray, normalize_obs(raw_obs))
                if normalize_obs is not None else raw_obs
            )
            probs, value, action = _policy_step(model_obj, policy_obs, mask, deterministic)

            raw.step_async(np.array([action], dtype=np.int64))
            obs, _rews, dones, infos = raw.step_wait()
            trace = np.asarray(raw.batch.get_trace(0), dtype=np.float32).reshape(-1, tf)
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
                final_score = float(cast(float, infos[0].get("score", score)))
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
    tiles = np.frombuffer(bytes(raw.batch.get_tiles(0)), dtype=np.uint8).reshape(grid_h, grid_w)
    return _Episode(tiles)


def _write_meta(f, model_path, config, episodes, deterministic,
                grid_w, grid_h, raw) -> None:
    rt, rp, rd, rw, og = raw.batch.reward_weights()
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
    dist_margin: float = typer.Option(
        0.10, "--dist-margin",
        help="Min extra normalised path-distance for a region to count as 'farther'."),
) -> None:
    """Print action frequencies and the near-vs-dense tradeoff the agent is making."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    chosen = []                  # chosen action per decision
    near_count, near_dist = [], []   # cluster_count / pathdist the agent committed to
    cluster_decisions = 0
    chose_sparse_near = 0   # picked a region while a denser one existed >margin farther

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
                if a < CLUSTER_K:                      # chose a region
                    cluster_decisions += 1
                    cc, cd = float(count[i, a]), float(pdist[i, a])
                    near_count.append(cc)
                    near_dist.append(cd)
                    # Did a denser region exist meaningfully farther than the one chosen?
                    # (the agent trading dense-far for sparse-near, now via region slots)
                    if ((count[i] > cc) & (pdist[i] > cd + dist_margin)).any():
                        chose_sparse_near += 1

    acts = np.concatenate(chosen) if chosen else np.empty(0, int)
    total = len(acts)
    log.info("\n=== %s — %d decisions over %d episodes ===", path.name, total, len(eps))
    log.info("\nAction frequency:")
    for a in range(len(names)):
        n = int((acts == a).sum())
        if n:
            log.info("  %-16s %6d  %5.1f%%", names[a], n, 100.0 * n / max(total, 1))

    if near_count:
        log.info("\nChosen-region decisions: %d  mean count=%.3f  mean pathdist=%.3f",
                 cluster_decisions, float(np.mean(near_count)), float(np.mean(near_dist)))
    if cluster_decisions:
        frac = 100.0 * chose_sparse_near / cluster_decisions
        log.info("  ...where a DENSER region existed >%.2f farther (chose sparse-near): %d (%.1f%%)",
                 dist_margin, chose_sparse_near, frac)
        log.info("  (high %s ⇒ the learned policy favours near-sparse over far-dense — raise "
                 "pickup weight or option_gamma to rebalance)", "%")


# ── view command ─────────────────────────────────────────────────────────────
#
# Renders, per high-level decision, four panels so reward / obs / action shaping
# can be debugged by eye:
#   • Grid       — obstacles, base, safezone, gold, items, agent; the chosen region
#                  is highlighted with an arrow to its obs path-nearest gold.
#   • Actions    — masked action-probability distribution (masked = hatched, chosen
#                  = outlined).
#   • Cluster obs — the 3×3 region grid the agent perceived (true count + count_norm
#                  + pathdist per region).
#   • Reward     — the per-tick reward breakdown the option earned + discounted return.
#
# matplotlib is imported lazily inside `view` so `record`/`summary` never pay for it.

_REGION_COLS = _REGION_ROWS = 3            # matches engine/clusters.rs (3×3 grid)
_CLUSTER_COUNT_NORM = 25.0                 # matches sim/src/engine/obs.rs


class _Trace:
    """Thin reader over one episode group of a trace.h5."""

    def __init__(self, f: h5py.File, ep_key: str) -> None:
        self.names = [n.decode() if isinstance(n, bytes) else str(n)
                      for n in f.attrs["action_names"]]
        self.gw = int(f.attrs["grid_w"])
        self.gh = int(f.attrs["grid_h"])
        self.rw = {k: float(f.attrs[k]) for k in f.attrs if k.startswith("reward_")}
        self.option_gamma = float(f.attrs["option_gamma"])
        g = f[ep_key]
        self.tiles = g["map_tiles"][:]                       # (H, W) uint8
        self.final_score = float(g.attrs["final_score"])
        self.base_x, self.base_y = int(g.attrs["base_x"]), int(g.attrs["base_y"])
        self.d = g["decisions"]
        self.n = int(g.attrs["n_decisions"])
        self.gold_off = self.d["gold_offsets"][:]
        self.item_off = self.d["item_offsets"][:]
        self.gold_xy = self.d["gold_xy"][:]
        self.item_xyk = self.d["item_xyk"][:]
        t = g["ticks"]                                       # per-tick reward rows
        self.t_dec = t["decision_idx"][:]
        self.t_r = {k: t[k][:] for k in ("r_tick", "r_pickup", "r_deposit", "r_wall",
                                         "r_total", "discount", "tick")}

    def gold_at(self, i: int) -> np.ndarray:
        return self.gold_xy[self.gold_off[i]:self.gold_off[i + 1]]

    def items_at(self, i: int) -> np.ndarray:
        return self.item_xyk[self.item_off[i]:self.item_off[i + 1]]

    def scalar(self, key: str, i: int):
        return self.d[key][i]


def _region_of(xs: np.ndarray, ys: np.ndarray, gw: int, gh: int) -> np.ndarray:
    rx = np.clip(xs * _REGION_COLS // max(gw, 1), 0, _REGION_COLS - 1)
    ry = np.clip(ys * _REGION_ROWS // max(gh, 1), 0, _REGION_ROWS - 1)
    return (ry * _REGION_COLS + rx).astype(int)


def _tile_rgb(tiles: np.ndarray) -> np.ndarray:
    """Map tile codes (0 free, 1 obstacle, 10+ base, 20+ safezone) → RGB image."""
    img = np.ones((*tiles.shape, 3), dtype=float)         # free = white
    img[tiles == 1] = (0.22, 0.22, 0.25)                  # obstacle = dark
    img[tiles >= 20] = (0.80, 0.95, 0.80)                 # safezone = light green
    img[(tiles >= 10) & (tiles < 20)] = (0.65, 0.80, 1.0)  # base = light blue
    return img


def _render(tr: _Trace, i: int, axes) -> None:
    """Draw the four panels for decision `i` into the provided (grid, act, clu, rew) axes."""
    import matplotlib.patches as mpatches
    ax_grid, ax_act, ax_clu, ax_rew = axes
    for a in axes:
        a.clear()

    n_regions = _REGION_COLS * _REGION_ROWS
    a = int(tr.scalar("chosen_action", i))
    ax, ay = int(tr.scalar("agent_x", i)), int(tr.scalar("agent_y", i))
    carried = int(tr.scalar("gold_carried", i))
    score = int(tr.scalar("score", i))
    value = float(tr.scalar("value", i))
    cdx = tr.scalar("cluster_dx", i)
    cdy = tr.scalar("cluster_dy", i)
    cpd = tr.scalar("cluster_pathdist", i)
    ccnt = tr.scalar("cluster_count", i)
    probs = tr.scalar("action_probs", i)
    mask = tr.scalar("action_mask", i).astype(bool)

    # ── Grid ────────────────────────────────────────────────────────────────
    ax_grid.imshow(_tile_rgb(tr.tiles), origin="upper", interpolation="nearest")
    for c in range(1, _REGION_COLS):
        ax_grid.axvline(tr.gw * c / _REGION_COLS - 0.5, color="0.6", lw=0.7, ls=":")
    for r in range(1, _REGION_ROWS):
        ax_grid.axhline(tr.gh * r / _REGION_ROWS - 0.5, color="0.6", lw=0.7, ls=":")
    gold = tr.gold_at(i)
    if len(gold):
        ax_grid.scatter(gold[:, 0], gold[:, 1], s=14, c="goldenrod",
                        edgecolors="k", linewidths=0.3, label="gold", zorder=3)
    items = tr.items_at(i)
    pu = items[items[:, 2] != 0] if len(items) else items   # non-gold = power-ups
    if len(pu):
        ax_grid.scatter(pu[:, 0], pu[:, 1], s=40, marker="*", c="magenta",
                        edgecolors="k", linewidths=0.3, label="item", zorder=3)
    ax_grid.scatter([tr.base_x], [tr.base_y], s=70, marker="s", c="blue",
                    edgecolors="k", label="base", zorder=4)
    ax_grid.scatter([ax], [ay], s=90, marker="o", c="red",
                    edgecolors="k", label="agent", zorder=5)
    # Highlight the chosen region + arrow to its obs-nearest gold.
    if a < n_regions:
        rx, ry = a % _REGION_COLS, a // _REGION_COLS
        x0, y0 = tr.gw * rx / _REGION_COLS - 0.5, tr.gh * ry / _REGION_ROWS - 0.5
        ax_grid.add_patch(mpatches.Rectangle(
            (x0, y0), tr.gw / _REGION_COLS, tr.gh / _REGION_ROWS,
            fill=False, edgecolor="red", lw=2.0, zorder=2))
        tx, ty = ax + cdx[a] * tr.gw, ay + cdy[a] * tr.gh
        ax_grid.annotate("", xy=(tx, ty), xytext=(ax, ay),
                         arrowprops=dict(arrowstyle="->", color="red", lw=1.5), zorder=6)
    act_name = tr.names[a] if a < len(tr.names) else f"act{a}"
    ax_grid.set_title(f"decision {i+1}/{tr.n}  tick={int(tr.scalar('tick', i))}\n"
                      f"action={act_name}  carried={carried}  score={score}  V={value:.2f}",
                      fontsize=9)
    ax_grid.set_xlim(-0.5, tr.gw - 0.5)
    ax_grid.set_ylim(tr.gh - 0.5, -0.5)
    ax_grid.legend(loc="upper right", fontsize=6, framealpha=0.8)
    ax_grid.set_xticks([])
    ax_grid.set_yticks([])

    # ── Action probabilities ──────────────────────────────────────────────────
    n_actions = len(probs)
    ypos = np.arange(n_actions)
    bars = ax_act.barh(ypos, probs,
                       color=["tab:blue" if k < n_regions else "tab:gray" for k in range(n_actions)])
    for k in range(n_actions):
        if not mask[k]:
            bars[k].set_hatch("xxx")
            bars[k].set_alpha(0.35)
    bars[a].set_edgecolor("red")
    bars[a].set_linewidth(2.0)
    ax_act.set_yticks(ypos)
    ax_act.set_yticklabels([tr.names[k] if k < len(tr.names) else str(k) for k in range(n_actions)],
                           fontsize=6)
    ax_act.invert_yaxis()
    ax_act.set_xlim(0, 1)
    ax_act.set_xlabel("policy prob", fontsize=7)
    ax_act.set_title("actions (hatched = masked, red = chosen)", fontsize=8)

    # ── Cluster obs (3×3) ───────────────────────────────────────────────────
    true_cnt = np.zeros(n_regions, int)
    if len(gold):
        for r in _region_of(gold[:, 0], gold[:, 1], tr.gw, tr.gh):
            true_cnt[r] += 1
    ax_clu.imshow(true_cnt.reshape(_REGION_ROWS, _REGION_COLS), cmap="YlOrBr", origin="upper")
    for k in range(n_regions):
        rx, ry = k % _REGION_COLS, k // _REGION_COLS
        ax_clu.add_patch(mpatches.Rectangle((rx - 0.5, ry - 0.5), 1, 1, fill=False,
                         edgecolor="red" if k == a else "0.7", lw=2.0 if k == a else 0.5))
        ax_clu.text(rx, ry - 0.22, f"n={true_cnt[k]}", ha="center", va="center", fontsize=6)
        ax_clu.text(rx, ry + 0.08, f"obs c={ccnt[k]:.2f}", ha="center", va="center", fontsize=5.5)
        ax_clu.text(rx, ry + 0.30, f"d={cpd[k]:.2f}", ha="center", va="center", fontsize=5.5)
    ax_clu.set_xticks([])
    ax_clu.set_yticks([])
    ax_clu.set_title("cluster obs: true count / obs count_norm / pathdist", fontsize=8)

    # ── Reward breakdown (this option) ────────────────────────────────────────
    sel = tr.t_dec == i
    comps = {"tick": tr.t_r["r_tick"][sel].sum(),
             "pickup": tr.t_r["r_pickup"][sel].sum(),
             "deposit": tr.t_r["r_deposit"][sel].sum(),
             "wall": tr.t_r["r_wall"][sel].sum()}
    total = float(tr.t_r["r_total"][sel].sum())
    ret = float(tr.scalar("option_return", i))
    oticks = int(tr.scalar("option_ticks", i))
    vals = list(comps.values()) + [total]
    bar_colors = ["tab:red" if v < 0 else "tab:green" for v in vals]
    bar_colors[-1] = "black"
    ax_rew.bar(list(comps) + ["TOTAL"], vals, color=bar_colors)
    ax_rew.axhline(0, color="k", lw=0.6)
    for j, v in enumerate(vals):
        ax_rew.text(j, v, f"{v:.2f}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=7)
    ax_rew.set_title(f"option reward  (ticks={oticks}, Σγᵗr={ret:.2f}, γ={tr.option_gamma})",
                     fontsize=8)
    ax_rew.tick_params(axis="x", labelsize=7)


@app.command()
def view(
    path: Path = typer.Argument(..., help="A trace.h5 produced by `atb-trace`."),
    episode: int = typer.Option(0, "--episode", "-e", help="Episode index to view."),
    decision: Optional[int] = typer.Option(
        None, "--decision", "-d",
        help="Show only this decision (else start interactive at decision 0)."),
    save: Optional[Path] = typer.Option(
        None, "--save",
        help="Headless: write one PNG per decision to this dir (no display)."),
) -> None:
    """Render a recorded trace — interactive slider, or PNG dump with --save.

    Four panels per decision: grid (agent/gold/items/obstacles), masked action
    probabilities, the 3×3 cluster observation, and the per-option reward breakdown.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    import matplotlib
    if save is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    with h5py.File(path, "r") as f:
        ep_key = f"episode_{episode:03d}"
        if ep_key not in f:
            eps = sorted(k for k in f if k.startswith("episode_"))
            raise typer.BadParameter(f"{ep_key} not found. Available: {eps}")
        tr = _Trace(f, ep_key)
        log.info("episode %d: %d decisions, final_score=%.1f, reward weights=%s",
                 episode, tr.n, tr.final_score, tr.rw)

        def make_axes():
            """Grid panel on the left; a 3-row right column (actions / cluster / reward)."""
            figure = plt.figure(figsize=(14, 8))
            gs = GridSpec(3, 2, width_ratios=[1.4, 1.0], figure=figure,
                          wspace=0.30, hspace=0.45, bottom=0.10)
            panels = (figure.add_subplot(gs[:, 0]), figure.add_subplot(gs[0, 1]),
                      figure.add_subplot(gs[1, 1]), figure.add_subplot(gs[2, 1]))
            return figure, panels

        if save is not None:                       # ── headless PNG dump ──
            save.mkdir(parents=True, exist_ok=True)
            idxs = [decision] if decision is not None else list(range(tr.n))
            for i in idxs:
                fig, axes = make_axes()
                _render(tr, i, axes)
                fig.savefig(save / f"ep{episode:03d}_dec{i:03d}.png", dpi=110,
                            bbox_inches="tight")
                plt.close(fig)
            log.info("wrote %d frame(s) → %s", len(idxs), save)
            return

        # ── interactive ──
        from matplotlib.backend_bases import Event, KeyEvent
        from matplotlib.widgets import Slider
        fig, axes = make_axes()
        start = decision if decision is not None else 0
        _render(tr, start, axes)

        sax = fig.add_axes((0.12, 0.02, 0.5, 0.03))
        slider = Slider(sax, "decision", 0, max(tr.n - 1, 0), valinit=start, valstep=1)

        def update(_val) -> None:
            _render(tr, int(slider.val), axes)
            fig.canvas.draw_idle()
        slider.on_changed(update)

        def on_key(event: Event) -> None:
            # mpl_connect types the callback as Callable[[Event], Any]; the
            # "key_press_event" always delivers a KeyEvent (which carries .key).
            key = cast(KeyEvent, event).key
            if key in ("right", "left"):
                step = 1 if key == "right" else -1
                slider.set_val(int(np.clip(slider.val + step, 0, tr.n - 1)))
        fig.canvas.mpl_connect("key_press_event", on_key)

        log.info("interactive — slider or ←/→ to scrub decisions; close window to exit.")
        plt.show()


if __name__ == "__main__":
    app()
