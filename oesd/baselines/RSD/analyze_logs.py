#!/usr/bin/env python3
"""
Analyze existing WandB logs to identify collapse patterns.
This script can help diagnose issues from past runs without re-running experiments.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def analyze_metrics_from_logs(log_dir):
    """
    Parse and analyze metrics from a WandB log directory.
    
    Args:
        log_dir: Path to wandb run directory
    """
    print(f"\n{'='*80}")
    print(f"Analyzing logs from: {log_dir}")
    print(f"{'='*80}\n")
    
    # Try to load wandb history
    try:
        import wandb
        api = wandb.Api()
        
        # Get run from log_dir
        runs = api.runs(path=None, filters={"state": "finished"})
        
        print("Found runs:")
        for run in runs[:10]:  # Show first 10
            print(f"  - {run.name} ({run.id}): {run.state}")
        
    except Exception as e:
        print(f"Could not load from WandB API: {e}")
        print("Will try to load from local files...")
    
    # Look for local metrics
    log_path = Path(log_dir)
    
    # Check for common metric patterns
    patterns_to_check = [
        'LossTe',
        'phi_obj',
        'direction_sim',
        'reward_g_distance',
        'DualCstPenalty',
        'LossSacp',
        'LossQf1',
        'LossQf2',
        'LossAlpha',
        'SZN/entropy',
        'SZN/V_z',
        'SZN/Regret',
    ]
    
    print("\nKey metrics to look for collapse:")
    print("-" * 80)
    print("\n1. SKILL GENERATOR COLLAPSE SIGNS:")
    print("   - SZN/entropy: Should be close to log(num_skills)")
    print("     * Low entropy → generator only samples few skills")
    print("   - Skill probabilities: Should be relatively uniform")
    print("     * One skill with >0.8 probability → collapse")
    
    print("\n2. ENCODER COLLAPSE SIGNS:")
    print("   - phi_obj: Should be non-zero and varying")
    print("     * Stuck at ~0 → encoder not learning")
    print("   - direction_sim: Should show variation")
    print("     * Constant value → encoder outputs not changing")
    
    print("\n3. POLICY COLLAPSE SIGNS:")
    print("   - LossSacp: Should decrease over time but not get stuck")
    print("     * Exploding/NaN → training instability")
    print("     * Stuck at constant value → policy not learning")
    print("   - reward_g_distance: Should vary and generally increase")
    print("     * Stuck at negative values → policy not reaching goals")
    
    print("\n4. DUAL CONSTRAINT ISSUES:")
    print("   - DualCstPenalty: Should be small (< dual_slack)")
    print("     * Large values → encoder violating smoothness constraint")
    print("   - DualLam: Should stabilize over time")
    print("     * Continuously increasing → encoder can't satisfy constraints")
    
    print("\n" + "=" * 80)
    print("RECOMMENDED ACTIONS BASED ON SYMPTOMS:")
    print("=" * 80)
    
    print("\nIf skills are all the same:")
    print("  1. Check SZN/entropy → if low, skill generator collapsed")
    print("     * Solution: Increase entropy regularization")
    print("     * Try increasing SZN_w3 (confidence penalty)")
    print("  2. Check phi_obj and direction_sim → if stuck, encoder collapsed")
    print("     * Solution: Reduce learning rate for encoder")
    print("     * Check dual_slack - might be too restrictive")
    print("  3. Check learned behavior visualizations")
    print("     * If trajectories are identical → policy collapsed")
    print("     * Solution: Check policy learning rate, reward_scale_factor")
    
    print("\nDebugging hyperparameters:")
    print("  - dual_slack: Controls encoder smoothness")
    print("    * Too small (< 1e-5): Encoder too constrained, can't learn diversity")
    print("    * Too large (> 1e-2): Encoder too free, violates Lipschitz constraint")
    print("  - SZN_w2: Controls KL with previous skills")  
    print("    * Too large: Skill generator can't explore new skills")
    print("  - SZN_w3: Controls confidence penalty (exploration)")
    print("    * Too small: Skill generator exploits same skills")
    print("    * Too large: Too much noise, unstable")
    
    return patterns_to_check


def plot_training_curves(metric_name='phi_obj'):
    """
    Example of how to plot training curves from saved data.
    """
    print(f"\nTo plot {metric_name} from your logs, use:")
    print(f"""
import wandb
api = wandb.Api()

# Get your run
run = api.run("YOUR_ENTITY/YOUR_PROJECT/RUN_ID")

# Get metric history
history = run.history()
if '{metric_name}' in history.columns:
    plt.figure(figsize=(12, 6))
    plt.plot(history['epoch'], history['{metric_name}'])
    plt.xlabel('Epoch')
    plt.ylabel('{metric_name}')
    plt.title('{metric_name} over time')
    plt.grid(True)
    plt.savefig('{metric_name}_curve.png')
    print("Saved plot to {metric_name}_curve.png")
    """)


def check_checkpoint_for_collapse(checkpoint_path):
    """
    Load a checkpoint and analyze the model state.
    
    Args:
        checkpoint_path: Path to .pt or .pkl checkpoint
    """
    import torch
    
    print(f"\nAnalyzing checkpoint: {checkpoint_path}")
    print("-" * 80)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print("\nCheckpoint contents:")
        for key in checkpoint.keys():
            if isinstance(checkpoint[key], dict):
                print(f"  {key}: {list(checkpoint[key].keys())[:5]}...")
            else:
                print(f"  {key}: {type(checkpoint[key])}")
        
        # Analyze specific components
        if 'SampleZPolicy' in checkpoint or 'skill_generator' in checkpoint:
            print("\n✓ Found skill generator in checkpoint")
            print("  To check for collapse, examine the output distribution")
            
        if 'traj_encoder' in checkpoint:
            print("\n✓ Found trajectory encoder in checkpoint")
            print("  To check for collapse, examine weight norms:")
            encoder_state = checkpoint['traj_encoder']
            
            # Check weight norms
            for name, param in encoder_state.items():
                if 'weight' in name and isinstance(param, torch.Tensor):
                    norm = torch.norm(param).item()
                    print(f"    {name}: norm={norm:.4f}")
                    
                    # Warning signs
                    if norm < 0.01:
                        print(f"      ⚠ Very small weights - might be dead/collapsed")
                    elif norm > 100:
                        print(f"      ⚠ Very large weights - might be unstable")
        
        if 'option_policy' in checkpoint:
            print("\n✓ Found policy in checkpoint")
            print("  To check for collapse, examine action distributions")
            
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Make sure the file exists and is a valid PyTorch checkpoint")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze RSD training logs to identify collapse'
    )
    parser.add_argument(
        '--log-dir', 
        type=str, 
        help='Path to WandB run directory'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Path to model checkpoint (.pt file)'
    )
    parser.add_argument(
        '--metric',
        type=str,
        default='phi_obj',
        help='Metric to analyze (default: phi_obj)'
    )
    
    args = parser.parse_args()
    
    if args.log_dir:
        analyze_metrics_from_logs(args.log_dir)
    elif args.checkpoint:
        check_checkpoint_for_collapse(args.checkpoint)
    else:
        print("RSD Collapse Analyzer")
        print("=" * 80)
        analyze_metrics_from_logs("./wandb")
        plot_training_curves(args.metric)
        
        print("\n" + "=" * 80)
        print("USAGE:")
        print("  python analyze_logs.py --log-dir /path/to/wandb/run")
        print("  python analyze_logs.py --checkpoint /path/to/checkpoint.pt")
        print("=" * 80)


if __name__ == '__main__':
    main()
