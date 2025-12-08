import os
from oesd.ours.unified_baseline_utils.SingleLoader import ModelConfig

VIS_PATH = "visualizations/"
CHECKPOINT_PATH = "baseline_checkpoints/"

SKILL_DIM = 8

model_cfgs = []

# model_cfgs.append(ModelConfig(
#     algo_name="DIAYN",
#     checkpoint_path=os.path.join(CHECKPOINT_PATH, "DIAYN/.."),
#     action_dim=None,
#     skill_dim=SKILL_DIM,
#     adapter_kwargs={"save_dir": os.path.join(VIS_PATH, "DIAYN")},
# ))

model_cfgs.append(ModelConfig(
    algo_name="LSD",
    checkpoint_path=os.path.join(CHECKPOINT_PATH, "LSD/lsd_latest.pth"),
    action_dim=None,
    skill_dim=SKILL_DIM,
    adapter_kwargs={"save_dir": os.path.join(VIS_PATH, "LSD")},
))

model_cfgs.append(ModelConfig(
    algo_name="RSD",
    checkpoint_path=os.path.join(CHECKPOINT_PATH, "RSD/4D_3P_img_dir_carry_orig_C/option_policy175.pt"),
    action_dim=None,
    skill_dim=SKILL_DIM,
    adapter_kwargs={"save_dir": os.path.join(VIS_PATH, "RSD")},
))

