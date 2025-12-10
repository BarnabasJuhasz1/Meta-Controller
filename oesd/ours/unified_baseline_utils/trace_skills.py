#!/usr/bin/env python3
"""
CLI script: Trace and visualize skill execution.

Examples:
    # Trace all skills for LSD (3 episodes each)
    python trace_skills.py --config configs/config1.py --algo LSD --episodes 3

    # Trace specific skills (0, 1, 2) for LSD
    python trace_skills.py --config configs/config1.py --algo LSD --skills 0,1,2

    # Trace skill 0 for both LSD and RSD
    python trace_skills.py --config configs/config1.py --algos LSD,RSD --skills 0

    # Output to custom directory
    python trace_skills.py --config configs/config1.py --algo LSD --output /path/to/output
"""

from __future__ import annotations
import argparse
import os
import sys

# Add oesd to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from SkillTracer import SkillTracer
from SingleLoader import load_model_from_config, load_config

from minigrid.envs import DoorKeyEnv
from oesd.ours.enviroments.example_minigrid import SimpleEnv


def build_env_factory(env_name: str = "minigrid", size: int = 8, max_steps: int = 200):
    """Create environment factory."""
    if env_name.lower() == "simpleenv" or env_name.lower() == "simple":
        def factory():
            return SimpleEnv(size=size, max_steps=max_steps, render_mode="rgb_array")
        return factory
    elif env_name.lower() == "minigrid" or env_name.lower() == "doorkey":
        def factory():
            return DoorKeyEnv(size=size, max_steps=max_steps, render_mode="rgb_array")
        return factory
    else:
        raise ValueError(f"Unknown env: {env_name}. Use 'simpleenv' or 'doorkey'")


def main():
    parser = argparse.ArgumentParser(
        description="Trace skill execution and generate visualizations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/config1.py",
        help="Config file with model definitions (e.g., configs/config1.py)",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default=None,
        help="Single algorithm to trace (e.g., LSD). If not provided, trace all in config.",
    )
    parser.add_argument(
        "--algos",
        type=str,
        default=None,
        help="Comma-separated list of algorithms (e.g., LSD,RSD). Overrides --algo.",
    )
    parser.add_argument(
        "--skills",
        type=str,
        default=None,
        help="Comma-separated skill indices to trace (e.g., 0,1,2). Default: all skills.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=2,
        help="Number of episodes per skill (default: 2)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=200,
        help="Max steps per episode (default: 200)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./skill_traces",
        help="Output directory for images (default: ./skill_traces)",
    )
    parser.add_argument(
        "--env_name",
        type=str,
        default="simpleenv",
        help="Environment name (default: simpleenv, or 'doorkey' for DoorKeyEnv)",
    )
    parser.add_argument(
        "--env_size",
        type=int,
        default=10,
        help="MiniGrid environment size (default: 10 for SimpleEnv, matching LSD training)",
    )

    args = parser.parse_args()

    # Load config and models
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)

    # Determine which algorithms to trace
    if args.algos:
        target_algos = [a.strip() for a in args.algos.split(",")]
    elif args.algo:
        target_algos = [args.algo]
    else:
        # Use all algos from config
        target_algos = [m.algo_name for m in config.model_cfgs]

    print(f"Target algorithms: {target_algos}")

    # Determine skills to trace
    target_skills = None
    if args.skills:
        target_skills = [int(s.strip()) for s in args.skills.split(",")]
        print(f"Target skills: {target_skills}")
    else:
        print("Target skills: ALL")

    # Build environment factory
    env_factory = build_env_factory(args.env_name, size=args.env_size, max_steps=args.horizon)

    # Import SkillRegistry
    from oesd.ours.unified_baseline_utils.skill_registry import SkillRegistry

    # Process each algorithm
    for algo_target in target_algos:
        # Find the model config
        model_cfg = None
        for m in config.model_cfgs:
            if m.algo_name.upper() == algo_target.upper():
                model_cfg = m
                break

        if model_cfg is None:
            print(f"[ERROR] Algorithm '{algo_target}' not found in config. Skipping.")
            continue

        print(f"\n{'='*70}")
        print(f"Loading model: {model_cfg.algo_name}")
        print(f"{'='*70}")

        # Load the adapter (with skill registry for adapters that need it)
        try:
            skill_count = getattr(model_cfg, 'skill_count', 8)  # Default to 8 if not specified
            skill_registry = SkillRegistry(skill_count)
            adapter = load_model_from_config(model_cfg, skill_registry=skill_registry)
            print(f"[OK] Model loaded: {model_cfg.algo_name} ({adapter.skill_dim} skills)")
        except Exception as e:
            print(f"[ERROR] Failed to load {model_cfg.algo_name}: {e}")
            continue

        # Create tracer
        algo_output_dir = os.path.join(args.output, model_cfg.algo_name)
        tracer = SkillTracer(
            env_factory=env_factory,
            adapter=adapter,
            algo_name=model_cfg.algo_name,
            output_dir=algo_output_dir,
        )

        # Trace skills
        if target_skills:
            # Trace specific skills
            for skill_idx in target_skills:
                if skill_idx >= adapter.skill_dim:
                    print(f"[WARN] Skill {skill_idx} >= {adapter.skill_dim} skills. Skipping.")
                    continue
                tracer.trace_skill(
                    skill_idx=skill_idx,
                    num_episodes=args.episodes,
                    save_images=True,
                    horizon=args.horizon,
                )
        else:
            # Trace all skills
            tracer.trace_all_skills(num_episodes=args.episodes, save_images=True)

    print(f"\n{'='*70}")
    print(f"[OK] All traces saved to: {args.output}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
