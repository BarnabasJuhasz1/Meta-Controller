#!/usr/bin/env python3
"""
Batch generate METRA-style plots for all configured baselines.

Usage:
    python batch_metra_plots.py --config oesd/ours/configs/config1.py --episodes 8 --horizon 200 --outdir ./metra_plots
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from oesd.ours.unified_baseline_utils.SingleLoader import load_config
from oesd.ours.unified_baseline_utils.metra_style_plot import main as run_metra_plot


def batch_generate(config_path: str, episodes: int, horizon: int, outdir: str):
    """Generate METRA-style plots for all algorithms in the config."""
    
    os.makedirs(outdir, exist_ok=True)
    
    config = load_config(config_path)
    
    # Get unique algorithm names
    algos = set()
    for m in config.model_cfgs:
        algos.add(m.algo_name.upper())
    
    print(f"Found {len(algos)} algorithms: {sorted(algos)}")
    
    for algo in sorted(algos):
        print(f"\n{'='*60}")
        print(f"Generating METRA-style plot for: {algo}")
        print(f"{'='*60}")
        
        outfile = os.path.join(outdir, f"metra_{algo.lower()}.png")
        
        # Build arguments for metra_style_plot
        args_list = [
            '--config', config_path,
            '--algo', algo,
            '--episodes', str(episodes),
            '--horizon', str(horizon),
            '--out', outfile,
            '--use_z'  # overlay Z basis arrows
        ]
        
        # Run the plot generation
        try:
            sys.argv = ['metra_style_plot.py'] + args_list
            run_metra_plot()
            print(f"✓ Saved: {outfile}")
        except Exception as e:
            print(f"✗ Failed for {algo}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"All plots saved to: {outdir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--episodes', type=int, default=8)
    parser.add_argument('--horizon', type=int, default=200)
    parser.add_argument('--outdir', type=str, default='./metra_plots')
    args = parser.parse_args()
    
    batch_generate(args.config, args.episodes, args.horizon, args.outdir)
