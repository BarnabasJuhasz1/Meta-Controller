# Utils/run_manager.py

import os
import json

def create_run_dir(base="LSD"):
    os.makedirs(base, exist_ok=True)

    # Find next run index
    existing = [d for d in os.listdir(base) if d.startswith("run")]
    if not existing:
        run_id = 1
    else:
        nums = [int(d.replace("run", "")) for d in existing]
        run_id = max(nums) + 1

    run_dir = os.path.join(base, f"run{run_id:03d}")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "skills"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "phi"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)

    return run_dir
