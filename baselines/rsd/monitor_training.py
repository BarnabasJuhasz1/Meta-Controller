#!/usr/bin/env python3
"""
Live monitoring script - attach to check metrics in real-time
Place near your training terminal and run periodically
"""

import sys
import time

def print_metric_guide():
    """Print a guide for interpreting training logs in real-time"""
    
    guide = """
╔════════════════════════════════════════════════════════════════════════════╗
║                      RSD LIVE TRAINING MONITOR GUIDE                       ║
╚════════════════════════════════════════════════════════════════════════════╝

When you see these patterns in your terminal output, here's what they mean:

┌─ SKILL GENERATOR COLLAPSE ─────────────────────────────────────────────────┐
│ Look for lines containing "SZN/"                                           │
│                                                                             │
│ ✓ HEALTHY:                                                                 │
│   SZN/entropy: ~2.08 (for 8 skills)                                        │
│   SZN/logp: varying between runs                                           │
│   mix_dist_prob: relatively balanced [0.1-0.2 each]                        │
│                                                                             │
│ ❌ COLLAPSE:                                                                │
│   SZN/entropy: < 1.0                                                       │
│   mix_dist_prob: one value > 0.8, others near 0                            │
│   → FIX: Increase SZN_w3, decrease SZN_w2                                  │
└─────────────────────────────────────────────────────────────────────────────┘

┌─ ENCODER COLLAPSE ─────────────────────────────────────────────────────────┐
│ Look for lines containing "phi_obj", "direction_sim", "DualCstPenalty"     │
│                                                                             │
│ ✓ HEALTHY:                                                                 │
│   phi_obj: non-zero, varying (e.g., 0.1 to 0.5)                           │
│   direction_sim: varying                                                   │
│   DualCstPenalty: < 1e-4 (your dual_slack)                                │
│                                                                             │
│ ❌ COLLAPSE:                                                                │
│   phi_obj: stuck at ~0.0                                                   │
│   DualCstPenalty: always = 1e-4 (hitting constraint)                      │
│   direction_sim: very small variance                                       │
│   → FIX: Increase dual_slack to 5e-3 or higher                            │
└─────────────────────────────────────────────────────────────────────────────┘

┌─ POLICY COLLAPSE ──────────────────────────────────────────────────────────┐
│ Look for lines containing "reward_g_distance", "LossSacp"                  │
│                                                                             │
│ ✓ HEALTHY:                                                                 │
│   reward_g_distance: varies, sometimes positive                            │
│   LossSacp: gradually decreasing                                           │
│   policy_rewards: non-zero                                                 │
│                                                                             │
│ ❌ COLLAPSE:                                                                │
│   reward_g_distance: always negative                                       │
│   LossSacp: stuck or not decreasing                                        │
│   → FIX: Check policy learning rate, reward scale                          │
└─────────────────────────────────────────────────────────────────────────────┘

┌─ TRAINING INSTABILITY ─────────────────────────────────────────────────────┐
│ Look for NaN, Inf, or exploding values                                     │
│                                                                             │
│ ❌ UNSTABLE:                                                                │
│   Any metric showing NaN or Inf                                            │
│   DualLam continuously growing (> 100)                                     │
│   Losses exploding (> 1000)                                                │
│   → FIX: Reduce all learning rates by 10x                                  │
└─────────────────────────────────────────────────────────────────────────────┘

╔════════════════════════════════════════════════════════════════════════════╗
║                        WHAT TO DO WHEN YOU SEE COLLAPSE                    ║
╚════════════════════════════════════════════════════════════════════════════╝

If collapse detected:
  1. Stop training (Ctrl+C in tmux)
  2. Identify which component from above
  3. Apply the suggested fix
  4. Restart with new hyperparameters

Your Current Settings (from run_rsd_minigrid.sh):
  dual_slack = 1e-4    ← LIKELY TOO SMALL
  SZN_w2 = 3           ← MODERATE
  SZN_w3 = 1           ← LIKELY TOO SMALL
  
Recommended Test:
  dual_slack = 5e-3    (increase 50x)
  SZN_w2 = 1           (decrease)
  SZN_w3 = 5           (increase 5x)

╔════════════════════════════════════════════════════════════════════════════╗
║                         GREP COMMANDS FOR QUICK CHECK                      ║
╚════════════════════════════════════════════════════════════════════════════╝

If your training prints to a log file, use these to quickly extract key metrics:

  # Check skill entropy
  grep "SZN/entropy" logfile.txt | tail -20
  
  # Check encoder objective
  grep "phi_obj" logfile.txt | tail -20
  
  # Check constraint penalty
  grep "DualCstPenalty" logfile.txt | tail -20
  
  # Check policy rewards
  grep "reward_g_distance" logfile.txt | tail -20

Or if running in tmux (your current setup):
  1. Ctrl+B, [ (enter copy mode)
  2. Ctrl+R to search backwards
  3. Type metric name (e.g., "SZN/entropy")
  4. Press 'n' to find next occurrence

╔════════════════════════════════════════════════════════════════════════════╗
║                              QUICK ACTIONS                                 ║
╚════════════════════════════════════════════════════════════════════════════╝

Stop training and try new hyperparameters:
  cd /home/juhasz/Desktop/UZH/Reinforcement_Learning/Project_31/Open-Ended-Skill-Discovery/baselines/RSD
  
  # Edit run_rsd_minigrid.sh
  # Change the last python command to:
  python tests/main.py \\
    --run_group RSD_exp --env minigrid --max_path_length 75 --seed 42 \\
    --traj_batch_size 128 --n_parallel 16 \\
    --n_epochs_per_eval 50 --n_epochs_per_save 25 --n_epochs_per_pt_save 25 \\
    --dim_option 8 --algo RSD \\
    --exp_name minigrid_FIXED_$(date +%Y%m%d_%H%M%S) \\
    --phi_type Projection --explore_type SZN \\
    --trans_optimization_epochs 30 \\
    --is_wandb 1 \\
    --SZN_w2 1 \\
    --SZN_w3 5 \\
    --SZN_window_size 5 \\
    --SZN_repeat_time 2 \\
    --n_epochs 1001 \\
    --Repr_max_step 75 \\
    --dual_slack 5e-3 \\
    --discrete 1
  
  # Then run:
  bash run_rsd_minigrid.sh

════════════════════════════════════════════════════════════════════════════════
"""
    print(guide)

if __name__ == "__main__":
    print_metric_guide()
    
    # Offer to watch mode
    if len(sys.argv) > 1 and sys.argv[1] == '--watch':
        print("\n[WATCH MODE] Refreshing every 30 seconds...")
        print("Press Ctrl+C to stop\n")
        try:
            while True:
                print("\n" + "="*80)
                print(f"Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                print("="*80)
                print_metric_guide()
                time.sleep(30)
        except KeyboardInterrupt:
            print("\nStopped watching.")
