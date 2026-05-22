# viewer/runner.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import pyray as rl

from .renderer import Renderer

# Speed steps: steps-per-second (0 = unlimited)
SPEEDS       = [0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 0.0]
SPEED_LABELS = ["0.5×", "1×", "2×", "5×", "10×", "30×", "max"]


def _find_config() -> str:
    here = Path(__file__).resolve()
    for parent in here.parents:
        c = parent / "assets" / "world" / "config.ron"
        if c.exists():
            return str(c)
    raise FileNotFoundError("Cannot find assets/world/config.ron")


def _load_policy(model_path: str | None):
    """Returns (model, vec_normalize | None) or None for random policy."""
    if model_path is None:
        return None
    from stable_baselines3 import PPO
    model = PPO.load(model_path)
    norm_path = Path(model_path).parent / "vecnormalize.pkl"
    vec_norm = None
    if norm_path.exists():
        import pickle
        with open(norm_path, "rb") as f:
            vec_norm = pickle.load(f)
    return model, vec_norm


def _pick_action(policy, obs_flat: list[float], obs_shape: tuple) -> int:
    if policy is None:
        import random
        return random.randint(0, 25)
    model, vec_norm = policy
    obs = np.array(obs_flat, dtype=np.float32).reshape(1, *obs_shape)
    if vec_norm is not None:
        obs = vec_norm.normalize_obs(obs)
    action, _ = model.predict(obs, deterministic=True)
    return int(action[0])


def run(model_path: str | None = None) -> None:
    import atb

    config    = _find_config()
    env       = atb.PyBatchEnv(1, config)
    obs_shape = atb.PyBatchEnv.obs_shape()
    policy    = _load_policy(model_path)

    gw, gh   = env.grid_size()
    renderer = Renderer(gw, gh)

    rl.set_config_flags(rl.FLAG_MSAA_4X_HINT)
    rl.init_window(renderer.win_w, renderer.win_h, "ATB — Agent Viewer")
    rl.set_target_fps(60)

    obs_list  = env.reset_all()
    speed_idx = 1       # start at 1× real-time
    paused    = False
    done      = False
    acc       = 0.0     # fractional step accumulator

    while not rl.window_should_close():
        dt = rl.get_frame_time()

        # ── input ─────────────────────────────────────────────────────────────
        if rl.is_key_pressed(rl.KEY_Q) or rl.is_key_pressed(rl.KEY_ESCAPE):
            break
        if rl.is_key_pressed(rl.KEY_R):
            obs_list = env.reset_all()
            done = False
            acc  = 0.0
        if rl.is_key_pressed(rl.KEY_SPACE):
            paused = not paused
        if rl.is_key_pressed(rl.KEY_RIGHT) or rl.is_key_pressed(rl.KEY_UP):
            speed_idx = min(speed_idx + 1, len(SPEEDS) - 1)
        if rl.is_key_pressed(rl.KEY_LEFT) or rl.is_key_pressed(rl.KEY_DOWN):
            speed_idx = max(speed_idx - 1, 0)

        # ── step ──────────────────────────────────────────────────────────────
        if not paused and not done:
            speed = SPEEDS[speed_idx]
            if speed == 0.0:
                steps = 30          # ~unlimited: drain a big batch each frame
            else:
                acc  += dt * speed
                steps = int(acc)
                acc  -= steps

            for _ in range(steps):
                action = _pick_action(policy, obs_list[0], obs_shape)
                obs_list, _rews, dones = env.step_batch([action])
                if dones[0]:
                    done = True
                    break

        # ── render ────────────────────────────────────────────────────────────
        tiles   = env.get_tiles(0)
        agents  = env.get_agents(0)
        items   = env.get_items(0)
        tick    = env.get_tick(0)
        m_ticks = env.get_match_ticks(0)

        rl.begin_drawing()
        rl.clear_background(rl.Color(20, 20, 24, 255))
        renderer.draw_world(tiles, agents, items, tick, m_ticks,
                            paused, SPEED_LABELS[speed_idx])
        if done:
            renderer.draw_end_screen(agents)
        rl.end_drawing()

    rl.close_window()
