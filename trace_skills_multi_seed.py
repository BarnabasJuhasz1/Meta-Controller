#!/usr/bin/env python3
"""
Run trace_skills.py with multiple seeds to generate skill traces across different environment layouts.

This script allows you to generate skill trajectory visualizations across multiple random seeds,
making it easier to cherry-pick the most interesting or representative visualizations.

Usage:
    # Generate traces for all algorithms with seeds 0-4
    python trace_skills_multi_seed.py --config configs/config1.py --seeds 0,1,2,3,4

    # Generate traces for specific algorithm with seeds 0-9
    python trace_skills_multi_seed.py --config configs/config1.py --algo LSD --seeds 0-9

    # Generate traces for multiple algorithms with custom output directory
    python trace_skills_multi_seed.py --config configs/config1.py --algos LSD,METRA --seeds 0-4 --output ./my_traces
"""

import argparse
import os
import subprocess
import sys


def parse_seed_range(seed_str: str) -> list[int]:
    """
    Parse seed string into list of integers.
    Supports formats like: "0,1,2,3" or "0-5"
    """
    if '-' in seed_str and ',' not in seed_str:
        # Range format: "0-5"
        start, end = seed_str.split('-')
        return list(range(int(start), int(end) + 1))
    else:
        # Comma-separated format: "0,1,2,3"
        return [int(s.strip()) for s in seed_str.split(',')]


def main():
    parser = argparse.ArgumentParser(
        description="Run trace_skills.py with multiple seeds.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/config1.py",
        help="Config file with model definitions (default: configs/config1.py)",
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
        help="Base output directory (default: ./skill_traces)",
    )
    parser.add_argument(
        "--env_name",
        type=str,
        default="simpleenv",
        help="Environment name (default: simpleenv)",
    )
    parser.add_argument(
        "--env_size",
        type=int,
        default=10,
        help="MiniGrid environment size (default: 10)",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0-4",
        help="Seeds to run (e.g., '0,1,2,3' or '0-9'). Default: 0-4",
    )

    args = parser.parse_args()

    # Parse seeds
    seeds = parse_seed_range(args.seeds)
    print(f"Running trace_skills.py with seeds: {seeds}")
    print(f"Output base directory: {args.output}")
    print("="*70)

    # Run trace_skills.py for each seed
    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"Running with seed {seed}")
        print(f"{'='*70}\n")

        # Create seed-specific output directory
        seed_output_dir = os.path.join(args.output, f"seed{seed}")

        # Build command
        cmd = [
            sys.executable,  # Use current Python interpreter
            "oesd/ours/unified_baseline_utils/trace_skills.py",
            "--config", args.config,
            "--episodes", str(args.episodes),
            "--horizon", str(args.horizon),
            "--output", seed_output_dir,
            "--env_name", args.env_name,
            "--env_size", str(args.env_size),
            "--seed", str(seed),
        ]

        # Add algo/algos if specified
        if args.algos:
            cmd.extend(["--algos", args.algos])
        elif args.algo:
            cmd.extend(["--algo", args.algo])

        # Add skills if specified
        if args.skills:
            cmd.extend(["--skills", args.skills])

        # Run the command
        try:
            result = subprocess.run(cmd, check=True)
            print(f"[OK] Seed {seed} completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Seed {seed} failed with return code {e.returncode}")
            continue

    print(f"\n{'='*70}")
    print(f"[OK] All seeds completed")
    print(f"Results saved to: {args.output}/seed0/, {args.output}/seed1/, ...")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
