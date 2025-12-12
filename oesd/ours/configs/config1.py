import os
from oesd.ours.unified_baseline_utils.SingleLoader import ModelConfig

VIS_PATH = "oesd/visualizations/"
CHECKPOINT_PATH = "checkpoints/"

SKILL_DIM = 8
ACTION_DIM = 6
# self._env.actions.forward,
# self._env.actions.left,
# self._env.actions.right,
# self._env.actions.pickup,
# self._env.actions.drop,
# self._env.actions.toggle,

COLORS = {
    "red": "#E22B34",  #diayn
    "gold": "#FFD300", #dads
    "pink": "#FF5CC0", #LSD
    "blue": "#438ADC", #metra
    "green": "#35C05E", #rsd
}

model_cfgs = []

model_cfgs.append(ModelConfig(
    algo_name="RSD",
    algo_color="green",
    checkpoint_path=os.path.join("oesd/baseline_checkpoints/RSD/4D_3P_img_dir_carry_orig_C/option_policy200.pt"),
    action_dim=ACTION_DIM,
    skill_dim=SKILL_DIM,
    adapter_kwargs={"save_dir": os.path.join(VIS_PATH, "RSD")},
))

model_cfgs.append(ModelConfig(
    algo_name="LSD",
    algo_color="purple",
    checkpoint_path="oesd/baseline_checkpoints/LSD/lsd_latest.pth",
    action_dim=ACTION_DIM,
    skill_dim=SKILL_DIM,
    adapter_kwargs={"save_dir": os.path.join(VIS_PATH, "LSD")},
))

model_cfgs.append(ModelConfig(
    algo_name="DIAYN",
    algo_color="red",
    checkpoint_path="oesd/baseline_checkpoints/DIAYN/diayn_doorkey.pth",
    action_dim=7,
    skill_dim=SKILL_DIM,
    adapter_kwargs={"save_dir": os.path.join(VIS_PATH, "DIAYN")},
))

model_cfgs.append(ModelConfig(
    algo_name="DADS",
    algo_color="pink",
    checkpoint_path="oesd/baseline_checkpoints/DADS/dads_doorkey.pth",
    action_dim=ACTION_DIM,
    skill_dim=SKILL_DIM,
    adapter_kwargs={"save_dir": os.path.join(VIS_PATH, "DADS")},
))

model_cfgs.append(ModelConfig(
    algo_name="METRA",
    algo_color="orange",
    checkpoint_path="oesd/baseline_checkpoints/METRA/",
    action_dim=7,
    skill_dim=SKILL_DIM,
    adapter_kwargs={"save_dir": os.path.join(VIS_PATH, "METRA")},
))