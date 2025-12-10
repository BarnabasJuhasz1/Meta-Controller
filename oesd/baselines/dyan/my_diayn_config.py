import os
from SingleLoader import ModelConfig

# Get the current directory of this file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Point to the checkpoint file
ckpt_path = os.path.join(current_dir, "diayn_doorkey.pth")

model_cfgs = [
    ModelConfig(
        algo_name="DIAYN",
        checkpoint_path=ckpt_path,
        action_dim=7,        # MiniGrid usually has 7 actions
        skill_dim=8,         # Placeholder, adapter uses registry count
        adapter_kwargs={"save_dir": current_dir}
    )
]