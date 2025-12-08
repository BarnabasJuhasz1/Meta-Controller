#!/usr/bin/env python3
# main.py
from __future__ import annotations

import os
import sys
import numpy as np
import gymnasium as gym

from SingleLoader import load_model_from_config, load_config
from SingleVisualizer import SingleVisualizer, VisualizerConfig, EnvConfig


# Only LSD is wired for training
from oesd.algorithms.lsd import LSDTrainer, LSDConfig

import argparse
import importlib

from minigrid.envs import DoorKeyEnv


# ============================================================================
# Helper Functions
# ============================================================================

# def ask(prompt: str, default=None, cast=str):
#     if default is not None:
#         prompt = f"{prompt} [{default}]: "
#     ans = input(prompt).strip()
#     if ans == "" and default is not None:
#         return cast(default)
#     return cast(ans)

# def ensure_dir(path: str):
#     os.makedirs(path, exist_ok=True)
#     return path


# ============================================================================
# Environment Selection Logic (SimpleEnv + Any Future Env)
# ============================================================================

def build_env_factory(env_id: str, max_steps: int = 200, render_mode: str = "rgb_array"):
    """
    Returns a factory() -> env function that works for:
        - "simple" (your SimpleEnv)
        - any future custom env names
        - any gym/gymnasium environment IDs
    """

    env_id = env_id.lower()

    # SIMPLE ENV SPECIAL CASE
    if env_id == "simple":
        from scripts.testing.example_minigrid import SimpleEnv
    

        def factory():
            return SimpleEnv(size=8, max_steps=max_steps, render_mode=render_mode)

        # we return a fake env to inspect action_dim
        tmp_env = factory()
        return factory, tmp_env
    elif env_id == "minigrid":

        def factory():
            return DoorKeyEnv(size=8, max_steps=max_steps, render_mode=render_mode)

        # we return a fake env to inspect action_dim
        tmp_env = factory()
        return factory, tmp_env
    # OTHERWISE → treat as gym environment ID
    else:
        def factory():
            return gym.make(env_id)

        tmp_env = gym.make(env_id)
        return factory, tmp_env



# ============================================================================
# Training API (Only LSD)
# ============================================================================

def train_lsd():
    print("\n=== LSD TRAINING ===")

    cfg = LSDConfig()
    episodes = ask("Number of episodes", default=str(cfg.num_episodes), cast=int)
    cfg.num_episodes = episodes

    trainer = LSDTrainer(cfg)
    trainer.train()

    SAVE_PATH = "checkpoints/lsd"
    os.makedirs(SAVE_PATH, exist_ok=True)

    trainer.save(SAVE_PATH)
    print("\nTraining complete.\n")



# ============================================================================
# Position Extraction (SimpleEnv + Any Future Env)
# ============================================================================

def unified_position_fn(env, obs, info):
    """
    Universal position extractor:
    - If env exposes agent_pos → use it
    - Else if tracking is present → use it
    - Else → return (0,0)
    """

    # Case 1 — environments with actual agent_pos (MiniGrid)
    if hasattr(env, "agent_pos"):
        try:
            pos = np.array(env.agent_pos, dtype=np.float32)
            if pos.shape == (2,):
                return pos
        except:
            pass

    # Case 2 — relative tracker for SimpleEnv or others
    # if hasattr(env, "_vis_x") and hasattr(env, "_vis_y"):
    #     return np.array([env._vis_x, env._vis_y], dtype=np.float32)

    # Case 3 — fallback
    return np.zeros(2, dtype=np.float32)



# ============================================================================
# Position Update Logic (called each step)
# This integrates relative motion for envs that do NOT expose agent_pos
# ============================================================================

# def update_relative_motion(env):
#     """
#     Fill in motion tracking for SimpleEnv or custom envs
#     using last action + agent_dir.
#     """

#     # Initialize state
#     if not hasattr(env, "_vis_initialized"):
#         env._vis_x = 0.0
#         env._vis_y = 0.0
#         env._vis_initialized = True

#     # Use MiniGrid-style orientation if present
#     agent_dir = getattr(env, "agent_dir", 0)
#     last_action = getattr(env, "_last_action", None)

#     # MiniGrid: action 2 = forward
#     if last_action == 2:
#         if agent_dir == 0:      # right
#             env._vis_x += 1
#         elif agent_dir == 1:    # down
#             env._vis_y += 1
#         elif agent_dir == 2:    # left
#             env._vis_x -= 1
#         elif agent_dir == 3:    # up
#             env._vis_y -= 1



parser = argparse.ArgumentParser(description="Unified Baseline Utils")
# parser.add_argument("--mode", type=str, default="visualize", choices=["train", "visualize"])
parser.add_argument("--algo_name", type=str, default="RSD", choices=["LSD", "RSD"])
parser.add_argument("--env_name", type=str, default="minigrid")
parser.add_argument("--horizon", type=int, default=200)
parser.add_argument("--episodes", type=int, default=3)
parser.add_argument("--deterministic", type=bool, default=False)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--render_mode", type=str, default="rgb_array")
parser.add_argument("--skill_idx", type=int, default=0)
parser.add_argument("--config", type=str, default="configs/config1.py")
parser.add_argument("--skill_count", type=int, default=8)


def main(_A: argparse.Namespace):

    # Build factory + temporary env
    env_factory, tmp_env = build_env_factory(_A.env_name)

    # load model configs from config file
    config = load_config(_A.config)

    # build visualizer config
    vis_cfg = VisualizerConfig(
        env=EnvConfig(factory=env_factory),
        models=config.model_cfgs,
        horizon=_A.horizon,
        num_episodes=_A.episodes,
        deterministic=_A.deterministic,
        seed=_A.seed,
        position_fn=unified_position_fn,
        skill_count=_A.skill_count,
    )
    # construct  SingleVisualizer 
    visualizer = SingleVisualizer(vis_cfg)

    # visualizer.update_relative_motion = update_relative_motion

    # trajectories = visualizer.sample_trajectories()

    # visualizer.plot_trajectories(trajectories)

    print("\nVisualization complete.\n")


if __name__ == "__main__":
    _A = parser.parse_args()
    main(_A)
