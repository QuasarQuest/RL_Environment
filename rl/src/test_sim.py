# rl/src/test_sim.py
#
# Verifies the full Rust->Python pipeline:
#   1. Env constructs and runs Startup
#   2. reset() returns a valid obs vector
#   3. step() advances one tick and returns (obs, reward, done)
#   4. A full episode runs to completion

import atb

def main():
    print(f"obs_dim={atb.PyRlEnv.obs_dim()}  action_size={atb.PyRlEnv.action_size()}")

    env = atb.PyRlEnv()

    obs = env.reset()
    assert len(obs) == atb.PyRlEnv.obs_dim(), f"Bad obs length: {len(obs)}"
    print(f"reset OK  obs[:5]={[round(v,3) for v in obs[:5]]}")

    total_reward = 0.0
    tick = 0
    done = False

    while not done:
        obs, reward, done = env.step(25)   # 25 = Wait
        total_reward += reward
        tick += 1
        if tick % 500 == 0:
            print(f"  tick={tick}  reward={reward:.4f}  total={total_reward:.2f}  done={done}")

    print(f"\nEpisode finished — ticks={tick}  total_reward={total_reward:.2f}")

    # Second episode — verify reset works after a full run
    obs2 = env.reset()
    assert len(obs2) == atb.PyRlEnv.obs_dim()
    print(f"reset #2 OK  obs[:5]={[round(v,3) for v in obs2[:5]]}")

if __name__ == "__main__":
    main()