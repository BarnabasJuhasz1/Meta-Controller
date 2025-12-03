#!/usr/bin/env python3
"""
Quick script to check WandB metrics for collapse without needing API access.
This prints guidance based on what metrics you should manually check.
"""

print("=" * 80)
print("QUICK DIAGNOSTICS: Check These Metrics in Your WandB Dashboard")
print("=" * 80)

print("\n📊 STEP 1: Open your WandB project in browser")
print("   Go to: https://wandb.ai/YOUR_ENTITY/YOUR_PROJECT")

print("\n🔍 STEP 2: Check Skill Generator Health")
print("   Chart to create: 'SZN/entropy' over time")
print("   ✓ HEALTHY:  entropy ~ 2.08 (for 8 skills)")
print("   ⚠ WARNING:  entropy between 1.0 - 1.5")
print("   ❌ COLLAPSE: entropy < 1.0")
print("\n   What it means:")
print("   - High entropy = using all skills roughly equally")
print("   - Low entropy = dominated by 1-2 skills")

print("\n🧠 STEP 3: Check Trajectory Encoder")  
print("   Charts to check:")
print("   1. 'phi_obj' - should fluctuate, not stuck at 0")
print("   2. 'direction_sim' - should vary over time")
print("   3. 'DualCstPenalty' - should be < 1e-4 (your dual_slack)")
print("\n   ✓ HEALTHY:  phi_obj varying, DualCstPenalty small")
print("   ⚠ WARNING:  phi_obj near 0, DualCstPenalty at dual_slack")
print("   ❌ COLLAPSE: phi_obj stuck at 0, very small direction_sim variance")

print("\n🎮 STEP 4: Check Policy")
print("   Charts to check:")
print("   1. 'reward_g_distance' - should sometimes be positive")
print("   2. 'LossSacp' - should generally decrease")
print("\n   ✓ HEALTHY:  reward_g_distance varies, sometimes positive")
print("   ⚠ WARNING:  reward_g_distance mostly negative")
print("   ❌ COLLAPSE: LossSacp not decreasing, reward always negative")

print("\n🖼️  STEP 5: Visual Inspection")
print("   Look at 'train_Maze_traj' images:")
print("   - Left panel: trajectories in environment")
print("   - Right panel: GMM/Z-space visualization")
print("\n   ✓ HEALTHY:  Trajectories go to different locations")
print("   ❌ COLLAPSE: All trajectories overlap")

print("\n📈 STEP 6: Check Skill Distribution Bar Charts")
print("   Look for images named '*-Regret.png'")
print("   These show probability over each skill")
print("\n   ✓ HEALTHY:  Bars roughly even (around 0.125 for 8 skills)")
print("   ❌ COLLAPSE: One bar dominates (> 0.8)")

print("\n" + "=" * 80)
print("SPECIFIC TO YOUR SETUP (from run_rsd_minigrid.sh)")
print("=" * 80)

current_settings = {
    "dim_option": 8,
    "dual_slack": 1e-4,
    "SZN_w2": 3,
    "SZN_w3": 1,
    "trans_optimization_epochs": 50,
}

print("\nYour current hyperparameters:")
for key, val in current_settings.items():
    print(f"  {key:30s} = {val}")

print("\n⚠️  POTENTIAL ISSUES WITH YOUR SETTINGS:")
print("   1. dual_slack=1e-4 is VERY restrictive")
print("      → May prevent encoder from learning diverse skills")
print("      → Try: 1e-3 or 5e-3")
print("\n   2. SZN_w3=1 is low")
print("      → Skill generator may not explore enough")
print("      → Try: 3 or 5")
print("\n   3. SZN_w2=3 with low SZN_w3")
print("      → Strong pull to previous skills, weak exploration")
print("      → Try balancing: SZN_w2=1, SZN_w3=3")

print("\n💡 RECOMMENDED TEST:")
print("   Try this combination:")
print("   --dual_slack 5e-3 --SZN_w2 1 --SZN_w3 5 --trans_optimization_epochs 30")

print("\n" + "=" * 80)
print("NEXT STEPS")
print("=" * 80)
print("\n1. Check the metrics above in your WandB dashboard")
print("2. If you see collapse, note which component (skill gen / encoder / policy)")
print("3. Try the suggested hyperparameter changes")
print("\nFor automated debugging, run:")
print("  python debug_collapse.py")
print("  (or integrate into RSD.py using DEBUGGING_GUIDE.md)")
print("\n" + "=" * 80)
