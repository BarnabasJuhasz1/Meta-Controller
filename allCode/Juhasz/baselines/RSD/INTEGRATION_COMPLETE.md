# Debugging Integration Complete! ✅

## What Was Added to RSD.py

I've successfully integrated the automated collapse debugging into `iod/RSD.py`. Here's what changed:

### 1. Initialization (Line 180)
Added debug counter initialization in `__init__`:
```python
# Initialize debugging counter
self._debug_epoch_counter = 0
```

### 2. Debugging Method (Line 886)
Added `_debug_collapse()` method that:
- Imports the CollapseDebugger
- Checks skill generator diversity (entropy, effective rank)
- Checks encoder embedding diversity
- Checks policy action diversity
- Automatically diagnoses which component is collapsing
- Logs results to WandB
- Prints diagnosis to console

### 3. Automated Calls (Lines 580-585)
Added debugging calls in `_train_components()` that run:
- Every 50 training epochs
- OR whenever the skill generator (SampleZPolicy) is updated
- Only when WandB is enabled

```python
# Run debugging periodically
if wandb.run is not None:
    self._debug_epoch_counter += 1
    
    if self._debug_epoch_counter % 50 == 0 or self.NumSampleTimes == 1:
        print(f"\n[DEBUG] Running collapse diagnostics...")
        self._debug_collapse(v, tensors)
```

## What You'll See During Training

### Console Output
Every 50 epochs (or when skills update), you'll see:

```
[DEBUG] Running collapse diagnostics (counter: 50)...

[DEBUG] Checking skill generator...
[DEBUG] Checking encoder...
[DEBUG] Checking policy...
[DEBUG] Diagnosing collapse...

================================================================================
COLLAPSE DIAGNOSIS:
================================================================================
✓ All components appear healthy!
  OR
⚠ Found 2 issue(s):
  1. Skill generator has low entropy (0.85/2.08). It's not exploring diverse skills.
  2. Encoder has very low effective rank (0.12). Encoder is collapsing - not producing diverse embeddings.
================================================================================
```

### WandB Logs
New metrics will appear in your WandB dashboard:

**Skill Generator Metrics:**
- `skill_generator/entropy` - Should be ~2.08 for 8 skills
- `skill_generator/max_skill_prob` - Should be < 0.3
- `skill_generator/effective_rank` - Higher is better
- `skill_generator/distribution` - Bar chart image

**Encoder Metrics:**
- `encoder/std_mean` - Should be > 0.1
- `encoder/pairwise_dist_mean` - Higher is better
- `encoder/effective_rank` - Higher is better
- `encoder/rank_ratio` - Should be > 0.5
- `encoder_between_skills/pairwise_dist_mean` - Different skills should have different encodings

**Policy Metrics:**
- `policy_actions/std_mean` - Action diversity
- `policy_between_skills/pairwise_dist_mean` - Should be > 0.5

**Diagnosis Flags:**
- `debug/skill_generator_healthy` - Boolean
- `debug/encoder_healthy` - Boolean  
- `debug/policy_healthy` - Boolean
- `debug/num_issues` - Count of detected issues

**Visualizations:**
- `skill_generator/distribution` - Probability distribution over skills
- `debug/` folder in WandB run directory contains detailed plots

## How to Use This

### Option 1: Use with Your New Hyperparameters (Recommended)
Your updated `run_rsd_minigrid.sh` already has the improved hyperparameters:
```bash
bash run_rsd_minigrid.sh
```

The debugging will run automatically and you'll see diagnostics every ~50 epochs.

### Option 2: Check Existing Run
If you want to stop your current run and restart with debugging:
```bash
# In your tmux session (Ctrl+B, then type):
# Ctrl+C to stop current training

# Then restart:
bash run_rsd_minigrid.sh
```

### Option 3: Customize Debug Frequency
Edit line 583 in `iod/RSD.py` to change frequency:
```python
# Currently: every 50 epochs
if self._debug_epoch_counter % 50 == 0 or self.NumSampleTimes == 1:

# For more frequent: every 10 epochs
if self._debug_epoch_counter % 10 == 0 or self.NumSampleTimes == 1:

# For less frequent: every 100 epochs
if self._debug_epoch_counter % 100 == 0 or self.NumSampleTimes == 1:
```

## Expected Benefits

With your new hyperparameters and debugging enabled:

1. **Early Detection**: You'll know within 100 epochs if there's a collapse issue
2. **Pinpoint the Problem**: Diagnosis tells you exactly which component is failing
3. **Visual Confirmation**: Plots show you what "healthy" vs "collapsed" looks like
4. **Historical Tracking**: WandB logs let you see when collapse started

## Troubleshooting

### "ImportError: No module named 'debug_collapse'"
- Make sure `debug_collapse.py` is in the `baselines/RSD/` directory (it is)
- The import is relative, so it should work automatically

### "Debug never runs"
- Check that `wandb.run is not None` (WandB must be enabled)
- Check that training reaches at least 50 epochs

### "Too much output"
- Debugging only runs every 50 epochs by default
- It's designed to not slow down training significantly
- Each debug run takes ~5-10 seconds

## What to Look For

Based on your hyperparameter changes, you should see:

**Good Signs:**
- `skill_generator/entropy` increases to ~2.0
- `skill_generator/max_skill_prob` stays below 0.3
- `encoder/rank_ratio` > 0.5
- No collapse warnings in diagnosis

**Warning Signs (Previous Settings):**
- `skill_generator/entropy` < 1.0 → increase SZN_w3
- `encoder/rank_ratio` < 0.3 → increase dual_slack
- Multiple collapse issues → check all hyperparameters

## Next Steps

1. ✅ Integration complete
2. ⏱️  Start/restart training with `bash run_rsd_minigrid.sh`
3. 👀 Watch console for "[DEBUG]" messages
4. 📊 Check WandB for new `debug/*` metrics
5. 🎯 Compare with previous runs to see improvement

The debugging is now fully integrated and will run automatically! You'll get detailed diagnostics showing exactly which component (if any) is collapsing.
