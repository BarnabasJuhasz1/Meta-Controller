# RSD Debugging Tools - README

## Problem
Your RSD skills are all becoming the same despite hyperparameter tuning. This could be due to collapse in one of three components:
1. **Skill Generator (SampleZPolicy)** - not producing diverse skills
2. **Trajectory Encoder** - not learning distinct representations
3. **Policy Network** - not differentiating behavior between skills

## Tools Created

### Quick Diagnosis Tools (No Code Changes)

1. **`quick_check.py`** - instant diagnosis guide
   ```bash
   python3 quick_check.py
   ```
   Shows what metrics to check in WandB and likely issues with your current hyperparameters.

2. **`monitor_training.py`** - live training monitor guide
   ```bash
   python3 monitor_training.py
   ```
   Shows patterns to watch for in your terminal/tmux output.

3. **`analyze_logs.py`** - historical analysis
   ```bash
   python3 analyze_logs.py --log-dir /path/to/wandb/run
   ```
   Analyzes past WandB runs to identify collapse patterns.

### Advanced Debugging Tools (Requires Code Integration)

4. **`debug_collapse.py`** - automated debugging module
   - Comprehensive diversity metrics
   - Automatic collapse detection
   - Visualizations and WandB logging
   - See `debugging_guide.md` for integration instructions

## Quick Start (5 minutes)

### Step 1: Run Quick Check
```bash
cd /home/juhasz/Desktop/UZH/Reinforcement_Learning/Project_31/Open-Ended-Skill-Discovery/baselines/RSD
python3 quick_check.py
```

### Step 2: Check Your WandB Dashboard

Open your WandB project and look for these key metrics:

- **SZN/entropy**: Should be ~2.08 for 8 skills
  - If < 1.0 → Skill generator collapsed
- **phi_obj**: Should vary, not stuck at 0
  - If stuck → Encoder collapsed
- **reward_g_distance**: Should sometimes be positive
  - If always negative → Policy collapsed

### Step 3: Identify the Issue

Based on Step 2, note which component is problematic.

### Step 4: Apply Fix

Your current hyperparameters:
```bash
--dual_slack 1e-4    # Controls encoder constraint (LIKELY TOO RESTRICTIVE)
--SZN_w2 3           # Controls exploitation  
--SZN_w3 1           # Controls exploration (LIKELY TOO LOW)
```

**Recommended fix** (based on analysis):
```bash
--dual_slack 5e-3    # Increased 50x to give encoder more freedom
--SZN_w2 1           # Decreased to reduce exploitation bias
--SZN_w3 5           # Increased 5x to encourage exploration
```

### Step 5: Test New Configuration

Edit `run_rsd_minigrid.sh` and replace the last python command with:

```bash
python tests/main.py \
  --run_group RSD_exp --env minigrid --max_path_length 75 --seed 42 \
  --traj_batch_size 128 --n_parallel 16 \
  --n_epochs_per_eval 50 --n_epochs_per_save 25 --n_epochs_per_pt_save 25 \
  --dim_option 8 --algo RSD \
  --exp_name minigrid_fixed_hyperparam \
  --phi_type Projection --explore_type SZN \
  --trans_optimization_epochs 30 \
  --is_wandb 1 \
  --SZN_w2 1 \
  --SZN_w3 5 \
  --SZN_window_size 5 \
  --SZN_repeat_time 2 \
  --n_epochs 1001 \
  --Repr_max_step 75 \
  --dual_slack 5e-3 \
  --discrete 1
```

Then run:
```bash
bash run_rsd_minigrid.sh
```

## Understanding the Metrics

### Skill Generator Health
```
SZN/entropy       - Uniformity of skill distribution
                    Healthy: ~log(num_skills) = 2.08 for 8 skills
                    Collapse: < 1.0

SZN/logp          - Log probability of sampled skills
                    Should vary between episodes

mix_dist_prob     - Probability distribution over skills
                    Healthy: relatively balanced
                    Collapse: one value > 0.8
```

### Encoder Health
```
phi_obj           - Encoder objective (direction similarity)
                    Healthy: varying, non-zero
                    Collapse: stuck at ~0

DualCstPenalty    - Lipschitz constraint violation
                    Healthy: < dual_slack
                    Collapse: = dual_slack (hitting limit)

direction_sim     - Trajectory-goal alignment
                    Healthy: varying
                    Collapse: very small, constant
```

### Policy Health
```
reward_g_distance - Goal-reaching reward
                    Healthy: sometimes positive, varying
                    Collapse: always negative

LossSacp          - Policy loss
                    Healthy: decreasing over time
                    Collapse: stuck or exploding
```

## Common Scenarios

### Scenario 1: "One skill dominates"
**Symptoms**: Bar charts show one skill with >80% probability
**Diagnosis**: Skill generator collapsed
**Fix**: Increase `SZN_w3` from 1 to 5

### Scenario 2: "Skills different but trajectories same"
**Symptoms**: Different skill IDs but identical behaviors
**Diagnosis**: Policy not responding to skills
**Fix**: Check encoder embeddings, increase `dual_slack`

### Scenario 3: "Everything looks random"
**Symptoms**: High entropy but no coherent behaviors
**Diagnosis**: Encoder not learning structure
**Fix**: Might be dual_slack too large, reduce to 1e-3

## Files Reference

```
baselines/RSD/
├── debug_collapse.py       - Automated debugging module
├── analyze_logs.py         - Historical log analysis
├── quick_check.py          - Quick diagnostic guide
├── monitor_training.py     - Live training monitor
├── debugging_guide.md      - Full integration guide (artifact)
└── debugging_summary.md    - Summary of tools (artifact)
```

## Integration (Optional, for Advanced Users)

To add automated debugging to your training:

1. Read `debugging_guide.md` (in artifacts)
2. Add the `_debug_collapse` method to `iod/RSD.py`
3. Call it periodically during training
4. Get detailed metrics logged to WandB automatically

## Troubleshooting

**Q: Training crashes with "module debug_collapse not found"**  
A: The debug module is optional. Only import it if you integrated the code.

**Q: Metrics still don't make sense**  
A: Run `analyze_logs.py` on a checkpoint to inspect model weights directly.

**Q: New hyperparameters don't help**  
A: Try even larger dual_slack (1e-2) and check if encoder is learning at all.

**Q: Where are the artifact files?**  
A: `debugging_guide.md` and `debugging_summary.md` are in the .gemini artifacts directory.

## Next Steps

1. ✅ Run `python3 quick_check.py` 
2. ✅ Check WandB metrics
3. ✅ Identify which component is collapsing
4. ✅ Try recommended hyperparameters
5. ⏱️  Monitor new training run
6. 📊 Compare results

Good luck! The key insight is that your `dual_slack=1e-4` is likely too restrictive and `SZN_w3=1` too low for exploration.
