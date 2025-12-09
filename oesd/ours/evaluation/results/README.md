# Meta Controller Evaluation Walkthrough

I have implemented the evaluation script `oesd/ours/evaluation/eval.py` and helper scripts to run it easily.

## Changes Created

### 1. Evaluation Script (`oesd/ours/evaluation/eval.py`)
This script loads the Meta Controller and runs evaluation episodes. It calculates:
- **Success Rate**: % of episodes where reward > 0.
- **Active Skill Ratio**: % of unique skills used.
- **Skill Entropy**: A measure of strategy diversity.
- **HRL Timeline**: Plots skill usage over time steps.
- **Skill Usage Histogram**: Plots frequency of each skill.

### 2. Run Scripts
- `oesd/ours/evaluation/run_eval.ps1` (for Windows PowerShell)
- `oesd/ours/evaluation/run_eval.sh` (for Bash)

## How to Run

1.  **Download your Model**: Place your trained model `.zip` file in `oesd/ours/train_results/checkpoints/`.
2.  **Update Script**:
    - Open `oesd/ours/evaluation/run_eval.ps1` (or `.sh`).
    - Edit the `$ModelPath` variable to match your specific filename.
3.  **Execute**:
    ```powershell
    # Windows
    .\oesd\ours\evaluation\run_eval.ps1
    ```
    ```bash
    # Linux/Mac
    ./oesd/ours/evaluation/run_eval.sh
    ```

## Outputs

Results will be saved in `oesd/ours/evaluation/results/`:
-   `results.json`: Raw metrics and logs.
-   `skill_usage.png`: Histogram of skill frequencies.
-   `hrl_timeline.png`: Step-by-step visualization of skill activation.

## Analysis Guide

The script prints an automatic analysis of the strategy:
-   **"Jack of All Trades"**: High entropy, low success. (Random skill usage)
-   **"Specialist"**: Low entropy, high success. (Found one good skill)
-   **"Diverse Solver"**: Moderate entropy, high success. (Using multiple skills effectively)

## Evaluation Results (Sample Run)
- **Success Rate**: 10.00%
- **Active Skill Ratio**: 0.12 (2/16 skills used)
- **Scenario**: Mixed / Learning in progress.

> [!NOTE]
> **RSD Workaround**: Since the RSD checkpoint was missing, I configured the evaluation to use the **DIAYN** model as a placeholder for the RSD slots (0-7). This allowed the evaluation to run, but skills 0-7 are effectively DIAYN skills now.
