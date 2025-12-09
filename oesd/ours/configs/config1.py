import os
from oesd.ours.unified_baseline_utils.SingleLoader import ModelConfig

VIS_PATH = "oesd/visualizations/"
CHECKPOINT_PATH = "oesd/baseline_checkpoints/"

SKILL_DIM = 8

model_cfgs = []

# LSD Config (Commented out)
# model_cfgs.append(ModelConfig(
#     algo_name="LSD",
#     checkpoint_path=os.path.join(CHECKPOINT_PATH, "LSD/.."),
#     action_dim=None,
#     skill_dim=SKILL_DIM,
#     adapter_kwargs={"save_dir": os.path.join(VIS_PATH, "LSD")},
# ))

# RSD Config
PATH = "/home/juhasz/Desktop/UZH/Reinforcement_Learning/Project_31/Open-Ended-Skill-Discovery/oesd/baselines/RSD/exp/RSD_small_exp/mini_4d_3P_img_dir_carry_original_Csd042_1764514102_minigrid_small_RSD/option_policy175.pt"
model_cfgs.append(ModelConfig(
    algo_name="RSD",
    #checkpoint_path=os.path.join(CHECKPOINT_PATH, "RSD/4D_3P_img_dir_carry_orig_C/itr_175.pkl"),
    checkpoint_path=PATH,
    action_dim=None,
    skill_dim=SKILL_DIM,
    adapter_kwargs={"save_dir": os.path.join(VIS_PATH, "RSD")},
))

# DIAYN Config
DIAYN_REPO_PATH = os.path.join("oesd", "baselines", "dyan", "diayn_doorkey.pth")
model_cfgs.append(ModelConfig(
    algo_name="DIAYN",
    checkpoint_path=DIAYN_REPO_PATH,
    action_dim=7,
    skill_dim=5,
    adapter_kwargs={"save_dir": os.path.join(VIS_PATH, "DIAYN")},
))
