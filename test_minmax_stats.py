#!/usr/bin/env python3
"""Quick script to compute and print obs min/max stats for LSD adapter."""

import sys
import os

# Add oesd to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "oesd"))

import numpy as np
from adapters.lsd_adapter import LSDAdapter

def main():
    checkpoint_path = "checkpoints/lsd/lsd_latest.pth"
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    adapter = LSDAdapter(action_dim=6, skill_dim=8)
    adapter.load_model(checkpoint_path)
    print("✓ Checkpoint loaded\n")
    
    print("Computing min/max statistics from 100 environment samples...")
    obs_min, obs_max = adapter.get_obs_minmax_stats(num_samples=1000)
    
    print(f"\nObservation vector statistics (147-dim, standardized):")
    print(f"  Global min: {obs_min.min():.6f}")
    print(f"  Global max: {obs_max.max():.6f}")
    print(f"  Range [min, max]: [{obs_min.min():.6f}, {obs_max.max():.6f}]")
    
    print(f"\nPer-dimension min values (first 10):")
    print(f"  {obs_min[:10]}")
    print(f"\nPer-dimension max values (first 10):")
    print(f"  {obs_max[:10]}")
    
    # Save stats for later use
    np.save("obs_min.npy", obs_min)
    np.save("obs_max.npy", obs_max)
    print(f"\n✓ Stats saved to obs_min.npy and obs_max.npy")
    
    # Test normalization on a sample obs
    print("\n" + "="*60)
    print("Testing min-max normalization on a sample observation:")
    print("="*60)
    obs = adapter.trainer._reset_env()
    obs_std = adapter.preprocess_observation(obs)
    obs_unit = adapter.normalize_obs_minmax(obs_std, obs_min, obs_max)
    
    print(f"Standardized obs range: [{obs_std.min():.6f}, {obs_std.max():.6f}]")
    print(f"Unit-normalized obs range: [{obs_unit.min():.6f}, {obs_unit.max():.6f}]")
    print(f"✓ All values in [0,1]? {(obs_unit.min() >= 0.0) and (obs_unit.max() <= 1.0)}")

if __name__ == "__main__":
    main()
