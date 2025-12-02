#!/usr/bin/env python3
# main.py
from __future__ import annotations

import os
import sys
from pathlib import Path
import numpy as np
import gymnasium as gym

from SingleLoader import load_model_from_config, ModelConfig
from SingleVisualizer import SingleVisualizer, VisualizerConfig, EnvConfig
from adapters.registry import ADAPTER_REGISTRY


# ============================================================================
# Utility helpers
# ============================================================================

def ask(prompt: str, default=None, cast=str):
    """Simple console input with optional default + casting."""
    if default is not None:
        prompt = f"{prompt} [{default}]: "
    ans = input(prompt).strip()
    if ans == "" and default is not None:
        return cast(default)
    return cast(ans)


def ensure_dir(path: str | Path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# ============================================================================
# ENV FACTORY (generic MiniGrid or any gym env) / wait on Dionigi
# ============================================================================

def make_env_factory(env_id: str, **env_kwargs):
    def factory():
        return gym.make(env_id, **env_kwargs)
    return factory


# ============================================================================
# TRAINING DISPATCH (each algorithm handles its own training mode)
# ============================================================================

def train_model(algo: str):
    """
    Training logic is algorithm-specific.

    For LSD: lsd.py already contains a full training entry via LSDTrainer.
    For DIAYN, CIC, etc., you will plug their training logic here.

    This function simply dispatches to the algorithm-specific training script.
    """
    algo = algo.lower()

    if algo == "lsd":
        print("\nLaunching LSD training script...")
        os.system("python lsd.py train")
        return

    # TODO: Add training dispatch for DIAYN, CIC, METRA etc.
    print(f"\nERROR: Training not implemented yet for '{algo}'.")
    sys.exit(1)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n=== Unified RL Interface ===\n")

    # ----------------------------------------------------------------------
    # 1) Mode selection
    # ----------------------------------------------------------------------
    mode = ask("Select mode: train / visualize", default="visualize").lower()
    if mode not in {"train", "visualize"}:
        print("Unknown mode. Must be 'train' or 'visualize'.")
        sys.exit(1)

    # ----------------------------------------------------------------------
    # MODE: TRAINING
    # ----------------------------------------------------------------------
    if mode == "train":
        print("\nAvailable algos:", ", ".join(ADAPTER_REGISTRY.keys()))
        algo = ask("Which algorithm to train?", default="lsd")

        train_model(algo)
        return

    # ----------------------------------------------------------------------
    # MODE: VISUALIZATION
    # ----------------------------------------------------------------------
    print("\n=== Visualization Mode ===")
    print("Available algos:", ", ".join(ADAPTER_REGISTRY.keys()))

    num_models = ask("How many models do you want to visualize?", default="1", cast=int)

    # ----------------------------------------------------------------------
    # ENVIRONMENT SELECTION
    # ----------------------------------------------------------------------
    env_id = ask("Environment ID", default="MiniGrid-Empty-8x8-v0")

    # infer action dimension
    tmp_env = gym.make(env_id)
    if not hasattr(tmp_env.action_space, "n"):
        print("\nERROR: This visualization pipeline only supports DISCRETE action spaces.\n")
        sys.exit(1)
    action_dim = tmp_env.action_space.n
    tmp_env.close()

    # ----------------------------------------------------------------------
    # MODEL SELECTION
    # ----------------------------------------------------------------------
    model_cfgs = []

    for i in range(num_models):
        print(f"\n--- Model {i+1} ---")

        algo = ask("Algorithm", default="lsd").lower()
        if algo not in ADAPTER_REGISTRY:
            print(f"Unknown algorithm '{algo}'")
            sys.exit(1)

        # Create algorithm-specific folders
        ckpt_folder = ensure_dir(f"checkpoints/{algo}")
        vis_folder = ensure_dir(f"visualizations/{algo}")

        print(f"Checkpoint folder: {ckpt_folder}")

        # Ask for checkpoint file
        ckpt = ask("Path to checkpoint", default=str(ckpt_folder / "latest.pth"))

        model_cfgs.append(
            ModelConfig(
                algo=algo,
                checkpoint_path=ckpt,
                action_dim=action_dim,
                skill_dim=8,  # all algos use 8 discrete skills
                adapter_kwargs={"save_dir": str(vis_folder)},  # optional
            )
        )

    # ----------------------------------------------------------------------
    # VISUALIZATION SETTINGS
    # ----------------------------------------------------------------------
    horizon = ask("Max steps per trajectory", default="200", cast=int)
    episodes = ask("Num episodes per model", default="5", cast=int)
    deterministic = ask("Deterministic actions? (0/1)", default="0", cast=int)
    seed = ask("Seed", default="0", cast=int)

    deterministic_flag = bool(deterministic)

    # ----------------------------------------------------------------------
    # BUILD VISUALIZER
    # ----------------------------------------------------------------------
    vis_cfg = VisualizerConfig(
        env=EnvConfig(factory=make_env_factory(env_id)),
        models=model_cfgs,
        horizon=horizon,
        num_episodes=episodes,
        deterministic=deterministic_flag,
        seed=seed,
    )

    print("\nLoading models...")
    visualizer = SingleVisualizer(vis_cfg)

    print("\nSampling trajectories...")
    trajectories = visualizer.sample_trajectories()

    print("\nPlotting trajectories...")
    visualizer.plot_trajectories(trajectories)

    print("\nVisualization complete.\n")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
