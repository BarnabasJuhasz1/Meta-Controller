#!/usr/bin/env python3
# main.py
from __future__ import annotations

import os
import sys
import numpy as np
import gymnasium as gym

from SingleLoader import load_model_from_config, ModelConfig
from SingleVisualizer import SingleVisualizer, VisualizerConfig, EnvConfig

# Only LSD is wired for training
from algorithms.lsd import LSDTrainer, LSDConfig


# ============================================================================
# Helper Functions
# ============================================================================

def ask(prompt: str, default=None, cast=str):
    if default is not None:
        prompt = f"{prompt} [{default}]: "
    ans = input(prompt).strip()
    if ans == "" and default is not None:
        return cast(default)
    return cast(ans)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


# ============================================================================
# Environment Selection Logic (SimpleEnv + Any Future Env)
# ============================================================================

def build_env_factory(env_id: str):
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
            return SimpleEnv(size=8, max_steps=200, render_mode="rgb_array")

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

    save_dir = ensure_dir("checkpoints/lsd")
    trainer.save(save_dir)
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
    if hasattr(env, "_vis_x") and hasattr(env, "_vis_y"):
        return np.array([env._vis_x, env._vis_y], dtype=np.float32)

    # Case 3 — fallback
    return np.zeros(2, dtype=np.float32)



# ============================================================================
# Position Update Logic (called each step)
# This integrates relative motion for envs that do NOT expose agent_pos
# ============================================================================

def update_relative_motion(env):
    """
    Fill in motion tracking for SimpleEnv or custom envs
    using last action + agent_dir.
    """

    # Initialize state
    if not hasattr(env, "_vis_initialized"):
        env._vis_x = 0.0
        env._vis_y = 0.0
        env._vis_initialized = True

    # Use MiniGrid-style orientation if present
    agent_dir = getattr(env, "agent_dir", 0)
    last_action = getattr(env, "_last_action", None)

    # MiniGrid: action 2 = forward
    if last_action == 2:
        if agent_dir == 0:      # right
            env._vis_x += 1
        elif agent_dir == 1:    # down
            env._vis_y += 1
        elif agent_dir == 2:    # left
            env._vis_x -= 1
        elif agent_dir == 3:    # up
            env._vis_y -= 1



# ============================================================================
# MAIN PROGRAM
# ============================================================================

def main():
    print("\n=== Unified RL Interface ===\n")

    # ----------------------------------------------------------------------
    # Choose mode
    # ----------------------------------------------------------------------
    mode = ask("Select mode: train / visualize", default="visualize")

    if mode == "train":
        algo = ask("Which algorithm to train?", default="lsd")
        if algo != "lsd":
            print("Only LSD training implemented.")
            return
        train_lsd()
        return


    # ----------------------------------------------------------------------
    # VISUALIZATION MODE
    # ----------------------------------------------------------------------
    print("\n=== Visualization Mode ===")

    num_models = ask("Number of models to visualize", default="1", cast=int)

    env_id = ask("Environment ID", default="simple")

    # Build factory + temporary env
    env_factory, tmp_env = build_env_factory(env_id)

    # Determine action_dim
    if not hasattr(tmp_env.action_space, "n"):
        print("ERROR: Only discrete action spaces supported.")
        sys.exit(1)

    action_dim = tmp_env.action_space.n
    tmp_env.close()

    # Collect model configs
    model_cfgs = []
    for i in range(num_models):
        print(f"\n--- Model {i+1} ---")
        algo = ask("Algorithm", default="lsd")

        ckpt_folder = ensure_dir(f"checkpoints/{algo}")
        vis_folder  = ensure_dir(f"visualizations/{algo}")
        ckpt = ask("Checkpoint path", default=f"{ckpt_folder}/latest.pth")

        model_cfgs.append(ModelConfig(
            algo=algo,
            checkpoint_path=ckpt,
            action_dim=action_dim,
            skill_dim=8,
            adapter_kwargs={"save_dir": vis_folder},
        ))

    # Visualization parameters
    horizon = ask("Max steps per trajectory", default="200", cast=int)
    episodes = ask("Num episodes per model", default="3", cast=int)
    deterministic = bool(ask("Deterministic actions? (0/1)", default="0", cast=int))
    seed = ask("Seed", default="0", cast=int)

    # ----------------------------------------------------------------------
    # Build visualizer configuration
    # ----------------------------------------------------------------------
    vis_cfg = VisualizerConfig(
        env=EnvConfig(factory=env_factory),
        models=model_cfgs,
        horizon=horizon,
        num_episodes=episodes,
        deterministic=deterministic,
        seed=seed,
        position_fn=unified_position_fn,
    )

    # ----------------------------------------------------------------------
    # Visualization
    # ----------------------------------------------------------------------
    print("\nLoading models...")
    visualizer = SingleVisualizer(vis_cfg)

    print("Sampling trajectories...")

    visualizer.update_relative_motion = update_relative_motion

    trajectories = visualizer.sample_trajectories()

    print("Plotting trajectories...")
    visualizer.plot_trajectories(trajectories)

    print("\nVisualization complete.\n")



if __name__ == "__main__":
    main()
