import os
from oesd.ours.unified_baseline_utils.SingleLoader import ModelConfig

VIS_PATH = "visualizations/"
CHECKPOINT_PATH = "baseline_checkpoints/"

SKILL_DIM = 8
ACTION_DIM = 6
# self._env.actions.forward,
# self._env.actions.left,
# self._env.actions.right,
# self._env.actions.pickup,
# self._env.actions.drop,
# self._env.actions.toggle,

model_cfgs = []

model_cfgs.append(ModelConfig(
    algo_name="RSD",
    algo_color="green",
    checkpoint_path=os.path.join(CHECKPOINT_PATH, "RSD/4D_3P_img_dir_carry_orig_C/option_policy175.pt"),
    action_dim=ACTION_DIM,
    skill_dim=SKILL_DIM,
    adapter_kwargs={"save_dir": os.path.join(VIS_PATH, "RSD")},
))

model_cfgs.append(ModelConfig(
    algo_name="LSD",
    algo_color="purple",
    checkpoint_path=os.path.join(CHECKPOINT_PATH, "LSD/lsd_latest.pth"),
    action_dim=ACTION_DIM,
    skill_dim=SKILL_DIM,
    adapter_kwargs={"save_dir": os.path.join(VIS_PATH, "LSD")},
))

model_cfgs.append(ModelConfig(
    algo_name="DIAYN",
    algo_color="red",
    checkpoint_path=os.path.join(CHECKPOINT_PATH, "DIAYN/diayn_doorkey.pth"),
    action_dim=7,
    skill_dim=SKILL_DIM,
    adapter_kwargs={"save_dir": os.path.join(VIS_PATH, "DIAYN")},
))

model_cfgs.append(ModelConfig(
    algo_name="DADS",
    algo_color="pink",
    checkpoint_path=os.path.join(CHECKPOINT_PATH, "DADS/dads_doorkey.pth"),
    action_dim=ACTION_DIM,
    skill_dim=SKILL_DIM,
    adapter_kwargs={"save_dir": os.path.join(VIS_PATH, "DADS")},
))

# model_cfgs.append(ModelConfig(
#     algo_name="METRA",
#     checkpoint_path=os.path.join(CHECKPOINT_PATH, "METRA/"),
#     action_dim=7,
#     skill_dim=SKILL_DIM,
#     adapter_kwargs={"save_dir": os.path.join(VIS_PATH, "METRA")},
# ))