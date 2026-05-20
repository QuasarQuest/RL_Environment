# rl/src/test_random_explore.py
#
# Sanity check: a uniformly random policy should occasionally find gold
# on a 20×20 map with 8 gold items and 500 ticks. If not, the task is too
# hard for PPO to bootstrap from random exploration and we need to either
# shrink the map, increase gold density, or shape reward more aggressively.
#
# Run with: python rl/src/test_random_explore.py

from pathlib import Path
import numpy as np
import atb


def _find_config() -> str:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "assets" / "world" / "config.ron"
        if candidate.exists():
            return str(candidate)
    raise RuntimeError(f"Cannot find config.ron from {here}")


def run_random_episode(env, action_size, rng):
    """Run one episode picking actions uniformly at random.
    Returns (total_reward, pickups, deliveries, ticks)."""
    env.reset()
    total = 0.0
    pickups = 0
    deliveries = 0
    ticks = 0
    done = False
    while not done:
        a = int(rng.integers(0, action_size))
        _, r, done = env.step(a)
        total += r
        ticks += 1
        # Reward decomposition:
        # tick penalty   = -0.001
        # pickup reward  = +0.5   → step reward ≈ +0.499
        # delivery (1g)  = +5.0   → step reward ≈ +4.999
        # delivery (Ng)  = +5N    → step reward ≈ +5N - 0.001
        # Pickups can stack (multiple gold tiles adjacent) — count by 0.5 buckets.
        if r > 1.0:
            # Delivery — number of gold delivered ≈ round((r + 0.001) / 5.0)
            n = round((r + 0.001) / 5.0)
            deliveries += max(n, 1)
        elif r > 0.1:
            # Pickup — number of gold ≈ round((r + 0.001) / 0.5)
            n = round((r + 0.001) / 0.5)
            pickups += max(n, 1)
    return total, pickups, deliveries, ticks


def main() -> None:
    config_path = _find_config()
    env = atb.PyRlEnv(config_path)
    action_size = atb.PyRlEnv.action_size()  # type: ignore[attr-defined]

    n_episodes = 30
    rng = np.random.default_rng(0)

    print(f"Running {n_episodes} random-policy episodes...")
    print(f"{'ep':>3}  {'reward':>8}  {'pickups':>7}  {'deliveries':>10}  ticks")

    totals = []
    pickups_per_ep = []
    deliveries_per_ep = []
    for i in range(n_episodes):
        total, picks, delivs, ticks = run_random_episode(env, action_size, rng)
        totals.append(total)
        pickups_per_ep.append(picks)
        deliveries_per_ep.append(delivs)
        print(f"{i:>3}  {total:>+8.3f}  {picks:>7d}  {delivs:>10d}  {ticks}")

    print()
    print(f"Mean reward     : {np.mean(totals):+.3f}")
    print(f"Mean pickups    : {np.mean(pickups_per_ep):.2f}")
    print(f"Mean deliveries : {np.mean(deliveries_per_ep):.2f}")
    print(f"% eps with pickup    : {100 * np.mean([p > 0 for p in pickups_per_ep]):.0f}%")
    print(f"% eps with delivery  : {100 * np.mean([d > 0 for d in deliveries_per_ep]):.0f}%")

    # Verdict
    print()
    pickup_rate = np.mean([p > 0 for p in pickups_per_ep])
    if pickup_rate < 0.3:
        print("⚠  Less than 30% of random episodes have ANY pickup.")
        print("   PPO will likely struggle to bootstrap. Consider:")
        print("   - smaller map (15×15)")
        print("   - more gold (item_density.gold up from 2.0 to 4.0)")
        print("   - longer episodes (match_duration_ticks up from 500)")
    elif pickup_rate < 0.7:
        print("✓ Pickup rate is workable but on the low side.")
        print("  Training should learn but may take a while to get off zero.")
    else:
        print("✓ Pickup rate is healthy — random exploration finds gold reliably.")
        print("  PPO should learn quickly from this.")


if __name__ == "__main__":
    main()
    