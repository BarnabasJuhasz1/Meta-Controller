#!/usr/bin/env python3
"""
CLI script: Trace and visualize skill execution.

Examples:
    # Trace all skills for LSD (3 episodes each) with seed 42
    python trace_skills.py --config oesd/ours/configs/config1.py --algo LSD --episodes 3 --seed 42

    # Trace specific skills (0, 1, 2) for LSD with seed 2
    python trace_skills.py --config oesd/ours/configs/config1.py --algo LSD --skills 0,1,2 --seed 2

    # Trace skill 0 for both LSD and RSD with custom seed
    python trace_skills.py --config oesd/ours/configs/config1.py --algos LSD,RSD --skills 0 --seed 10

    # Output to custom directory
    python trace_skills.py --config oesd/ours/configs/config1.py --algo LSD --output /path/to/output --seed 42
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


def generate_color_shades(base_color: str, num_shades: int) -> list[str]:
    """
    Generate shades from dark to light based on a base hex color.
    First skill is darkest, last skill is lightest (almost white).
    
    Args:
        base_color: hex color like "#E22B34"
        num_shades: number of shades to generate
        
    Returns:
        list of hex color strings
    """
    # Parse hex color
    base_color = base_color.lstrip('#')
    r, g, b = int(base_color[0:2], 16), int(base_color[2:4], 16), int(base_color[4:6], 16)
    
    if num_shades == 1:
        return [f"#{base_color}"]
    
    # Generate more intense, vivid shades from dark to bright
    shades = []
    for i in range(num_shades):
        # Progress from 0 to 1 (darkest to lightest)
        t = i / (num_shades - 1)
        
        # Start dark (50% of base color), end bright (120% of base, capped at 255)
        # This creates more vibrant, popping colors
        dark_r, dark_g, dark_b = int(r * 0.5), int(g * 0.5), int(b * 0.5)
        light_r = min(255, int(r * 1.2))
        light_g = min(255, int(g * 1.2))
        light_b = min(255, int(b * 1.2))
        
        # Interpolate between dark and bright
        new_r = int(dark_r + (light_r - dark_r) * t)
        new_g = int(dark_g + (light_g - dark_g) * t)
        new_b = int(dark_b + (light_b - dark_b) * t)
        
        shades.append(f"#{new_r:02x}{new_g:02x}{new_b:02x}")
    
    return shades


def build_env_factory(env_name: str = "minigrid", size: int = 8, max_steps: int = 200, env_seed: int = 42):
    """Create environment factory with specific seed."""
    if env_name.lower() == "simpleenv" or env_name.lower() == "simple":
        def factory():
            env = SimpleEnv(size=size, max_steps=max_steps, render_mode="rgb_array")
            env.reset(seed=env_seed)
            return env
        return factory
    elif env_name.lower() == "minigrid" or env_name.lower() == "doorkey":
        def factory():
            env = DoorKeyEnv(size=size, max_steps=max_steps, render_mode="rgb_array")
            env.reset(seed=env_seed)
            return env
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
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Environment seed for consistent layouts (default: 42)",
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
    env_factory = build_env_factory(args.env_name, size=args.env_size, max_steps=args.horizon, env_seed=args.seed)
    
    # Get color mapping from config if available
    color_map = {}
    if hasattr(config, 'COLORS'):
        # Map algo names to colors
        algo_to_color_name = {
            'RSD': 'green',
            'LSD': 'pink',
            'DIAYN': 'red',
            'DADS': 'gold',
            'METRA': 'blue'
        }
        
        for algo_name, color_name in algo_to_color_name.items():
            if color_name in config.COLORS:
                color_map[algo_name] = config.COLORS[color_name]
        
        print(f"Using colors from config: {color_map}")

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

        # Get base color for this algorithm
        base_color = color_map.get(model_cfg.algo_name.upper(), "#1f77b4")  # Default to blue
        
        # Generate color shades for each skill
        skill_colors = generate_color_shades(base_color, skill_count)
        
        # Create tracer
        algo_output_dir = os.path.join(args.output, model_cfg.algo_name)
        tracer = SkillTracer(
            env_factory=env_factory,
            adapter=adapter,
            algo_name=model_cfg.algo_name,
            output_dir=algo_output_dir,
            skill_colors=skill_colors,
        )

        # Trace skills
        if target_skills:
            # Trace specific skills individually
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
            # Trace all skills in one combined image
            tracer.trace_all_skills_combined(
                num_episodes=args.episodes,
                save_images=True,
                seed=args.seed,
                horizon=args.horizon,
            )

    print(f"\n{'='*70}")
    print(f"[OK] All traces saved to: {args.output}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
