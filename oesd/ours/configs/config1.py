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

# RSD Config (MOCKED with DIAYN because RSD checkpoint is missing)
DIAYN_REPO_PATH = os.path.join("oesd", "baselines", "dyan", "diayn_doorkey.pth")
model_cfgs.append(ModelConfig(
    algo_name="DIAYN_PLACEHOLDER",
    checkpoint_path=DIAYN_REPO_PATH,
    action_dim=7,
    skill_dim=8,
    adapter_kwargs={"save_dir": os.path.join(VIS_PATH, "RSD_MOCK")},
))

# DIAYN Config
DIAYN_REPO_PATH = os.path.join("oesd", "baselines", "dyan", "diayn_doorkey.pth")
model_cfgs.append(ModelConfig(
    algo_name="DIAYN",
    checkpoint_path=DIAYN_REPO_PATH,
    action_dim=7,
    skill_dim=8,
    adapter_kwargs={"save_dir": os.path.join(VIS_PATH, "DIAYN")},
))
