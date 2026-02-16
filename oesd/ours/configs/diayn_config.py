import os
from oesd.ours.unified_baseline_utils.SingleLoader import ModelConfig

VIS_PATH = "oesd/visualizations/"
SKILL_DIM = 8

model_cfgs = []

model_cfgs.append(ModelConfig(
    algo_name="DIAYN",
    checkpoint_path="oesd/baselines/dyan/diayn_doorkey.pth",
    action_dim=7,
    skill_dim=SKILL_DIM,
    adapter_kwargs={"save_dir": os.path.join(VIS_PATH, "DIAYN")},
))
