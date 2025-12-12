#!/usr/bin/env python3
"""
Generate heatmaps across multiple seeds for cherry-picking the best visualizations.

For each seed, creates a subdirectory with heatmaps for all algorithms.
Outputs to: ./heatmaps/seed0/, ./heatmaps/seed1/, etc.
"""

import os
import sys
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description='Generate heatmaps across multiple seeds')
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(10)),
                        help='List of seeds to try (default: 0-9)')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of episodes per skill')
    parser.add_argument('--horizon', type=int, default=250,
                        help='Max steps per episode')
    parser.add_argument('--outdir', type=str, default='./heatmaps2',
                        help='Base output directory for all seeds')
    parser.add_argument('--config', type=str, default='oesd/ours/configs/config1.py',
                        help='Path to config file')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("GENERATING HEATMAPS ACROSS MULTIPLE SEEDS")
    print("="*70) 
    print(f"Seeds: {args.seeds}")
    print(f"Episodes per skill: {args.episodes}")
    print(f"Horizon: {args.horizon}")
    print(f"Base output directory: {args.outdir}")
    print()
    
    # Create base output directory
    os.makedirs(args.outdir, exist_ok=True)
    
    for seed in args.seeds:
        print(f"\n{'='*70}")
        print(f"SEED {seed}")
        print(f"{'='*70}\n")
        
        # Create seed-specific output directory
        seed_outdir = os.path.join(args.outdir, f"seed{seed}")
        os.makedirs(seed_outdir, exist_ok=True)
        
        # Call generate_heatmaps.py with specific seed
        cmd = [
            sys.executable,
            "generate_heatmaps.py",
            "--config", args.config,
            "--episodes", str(args.episodes),
            "--horizon", str(args.horizon),
            "--outdir", seed_outdir,
            "--seed", str(seed)
        ]
        
        print(f"Running: {' '.join(cmd)}\n")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Failed for seed {seed}: {e}")
            continue
        
        print(f"\n✓ Completed seed {seed}")
    
    print(f"\n{'='*70}")
    print("✓ ALL SEEDS COMPLETED")
    print(f"{'='*70}")
    print(f"\nResults saved in subdirectories:")
    for seed in args.seeds:
        seed_dir = os.path.join(args.outdir, f"seed{seed}")
        if os.path.exists(seed_dir):
            print(f"  - {seed_dir}")
    print()

if __name__ == '__main__':
    main()
