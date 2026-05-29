#!/usr/bin/env python3
"""Benchmark BT and GOAP scripted-policy scores to establish RL training targets.

Runs N full episodes for each policy using the stage config and reports
min / median / mean / max score (gold deposited).  The RL agent should
comfortably beat the GOAP median and approach or exceed the BT median.

Usage
-----
    python rl/scripts/benchmark_baselines.py
    python rl/scripts/benchmark_baselines.py --n 50 --stage 1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
RL_DIR     = SCRIPT_DIR.parent
PROJECT_ROOT = RL_DIR.parent
sys.path.insert(0, str(RL_DIR / "src"))

# ── Constants (must match sim/src/config.rs and rl/action.rs) ─────────────────
AGENT_MAX_GOLD          = 5
CLUSTER_K               = 4
ACTION_NAVIGATE_TO_BASE = 4
ACTION_WAIT             = 10
RADII                   = [3, 5, 8, 12, 20, 50]


# ── Clustering (mirrors clusters.rs) ──────────────────────────────────────────

def _chebyshev(ax, ay, bx, by):
    return max(abs(ax - bx), abs(ay - by))


def _find(parent, x):
    root = x
    while parent[root] != root:
        root = parent[root]
    while parent[x] != root:
        parent[x], x = root, parent[x]
    return root


def _union(parent, rank, i, j):
    ri, rj = _find(parent, i), _find(parent, j)
    if ri == rj:
        return
    if rank[ri] < rank[rj]:
        ri, rj = rj, ri
    parent[rj] = ri
    if rank[ri] == rank[rj]:
        rank[ri] += 1


def find_clusters(gold_xy: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    """Return up to CLUSTER_K clusters sorted by centroid (x, y).
    Each cluster is a list of (x, y) gold positions.
    """
    result: list[list[tuple[int, int]]] = [[] for _ in range(CLUSTER_K)]
    if not gold_xy:
        return result
    n = len(gold_xy)

    for radius in RADII:
        parent = list(range(n))
        rank   = [0] * n
        for i in range(n):
            ax, ay = gold_xy[i]
            for j in range(i + 1, n):
                bx, by = gold_xy[j]
                if _chebyshev(ax, ay, bx, by) <= radius:
                    _union(parent, rank, i, j)

        groups: dict[int, list[int]] = {}
        for i in range(n):
            groups.setdefault(_find(parent, i), []).append(i)

        if len(groups) <= CLUSTER_K:
            clusters = []
            for indices in groups.values():
                golds = [gold_xy[i] for i in indices]
                cx = sum(g[0] for g in golds) / len(golds)
                cy = sum(g[1] for g in golds) / len(golds)
                clusters.append((cx, cy, golds))
            clusters.sort(key=lambda c: (c[0], c[1]))
            for k, (_, _, golds) in enumerate(clusters):
                result[k] = golds
            return result

    result[0] = list(gold_xy)
    return result


# ── Policy helpers ─────────────────────────────────────────────────────────────

def _nearest_cluster_action(ax: int, ay: int, items) -> int:
    """Return the cluster action index (0-3) whose nearest gold is closest."""
    gold_xy = [(x, y) for x, y, kind in items if kind == 0]
    if not gold_xy:
        return ACTION_WAIT
    clusters = find_clusters(gold_xy)
    best_k, best_dist = None, float("inf")
    for k, cluster in enumerate(clusters):
        if not cluster:
            continue
        d = min(_chebyshev(ax, ay, gx, gy) for gx, gy in cluster)
        if d < best_dist:
            best_dist, best_k = d, k
    return best_k if best_k is not None else ACTION_WAIT


def bt_policy(agent, items) -> int:
    """BT: full inventory → base, else → nearest gold cluster."""
    ax, ay, _team, gold_carried, _score = agent
    if gold_carried >= AGENT_MAX_GOLD:
        return ACTION_NAVIGATE_TO_BASE
    return _nearest_cluster_action(ax, ay, items)


def goap_policy(agent, items) -> int:
    """GOAP for stage 1 (no enemies, no health items) — plan reduces to BT."""
    ax, ay, _team, gold_carried, _score = agent
    if gold_carried >= AGENT_MAX_GOLD:
        return ACTION_NAVIGATE_TO_BASE
    return _nearest_cluster_action(ax, ay, items)


# ── Benchmark runner ───────────────────────────────────────────────────────────

def _run_benchmark(config_path: str, n_episodes: int, policy_fn, label: str) -> np.ndarray:
    import atb  # noqa: PLC0415

    env    = atb.PyBatchEnv(1, config_path)
    scores = []

    for ep in range(n_episodes):
        env.reset_all()
        match_ticks = env.get_match_ticks(0)

        for _ in range(match_ticks):
            agents = env.get_agents(0)
            items  = env.get_items(0)
            action = policy_fn(agents[0], items)
            _, _, dones = env.step_batch([action])
            if dones[0]:
                break

        score = env.get_agents(0)[0][4]  # a.score
        scores.append(score)
        print(f"  {label}  ep {ep + 1:>3}/{n_episodes}  score={score}", flush=True)

    arr = np.array(scores, dtype=float)
    print(f"\n  {label}  n={n_episodes}  "
          f"min={arr.min():.0f}  median={np.median(arr):.0f}  "
          f"mean={arr.mean():.1f}  max={arr.max():.0f}")
    return arr


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n",     type=int, default=30, help="Episodes per policy (default 30)")
    parser.add_argument("--stage", type=int, default=1,  help="Stage config to use (default 1)")
    args = parser.parse_args()

    config_path = PROJECT_ROOT / "assets" / "world" / f"config_stage{args.stage}.ron"
    if not config_path.exists():
        sys.exit(f"[ERROR] Config not found: {config_path}")

    print(f"Config : {config_path}")
    print(f"Episodes per policy: {args.n}\n")

    print("─── BT ────────────────────────────────────────────────")
    bt_scores = _run_benchmark(str(config_path), args.n, bt_policy, "BT  ")
    print()
    print("─── GOAP ──────────────────────────────────────────────")
    goap_scores = _run_benchmark(str(config_path), args.n, goap_policy, "GOAP")
    print()
    print("═══ Summary ════════════════════════════════════════════")
    print(f"  BT   min={bt_scores.min():.0f}  median={np.median(bt_scores):.0f}  "
          f"max={bt_scores.max():.0f}")
    print(f"  GOAP min={goap_scores.min():.0f}  median={np.median(goap_scores):.0f}  "
          f"max={goap_scores.max():.0f}")
    print()
    print("  RL target: median > BT median")


if __name__ == "__main__":
    main()
