from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import CheckpointCallback

from oesd.ours.unified_baseline_utils.skill_registry import SkillRegistry
from oesd.ours.contoller.meta_env_wrapper import MetaControllerEnv
from oesd.ours.unified_baseline_utils.SingleLoader import load_model_from_config, load_config

import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument("--env_name", type=str, default="minigrid")
parser.add_argument("--skill_count_per_algo", type=int, default=10)
parser.add_argument("--skill_duration", type=int, default=10)
parser.add_argument("--num_timesteps", type=int, default=100000)
parser.add_argument("--learning_rate", type=float, default=3e-4)
parser.add_argument("--n_steps", type=int, default=2048)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--verbose", type=int, default=1)
parser.add_argument("--tensorboard_log", type=str, default="ours/train_results/logs/")
parser.add_argument("--save_path", type=str, default="ours/train_results/")
parser.add_argument("--config_path", type=str, default="ours/configs/config1.py")
parser.add_argument("--checkpoint_freq", type=int, default=20, help="Save model every k epochs")
parser.add_argument("--render-mode", type=str, default="rgb_array")
parser.add_argument("--key_pickup_reward", type=float, default=0.1, help="Reward for picking up the key (once per episode)")
parser.add_argument("--door_open_reward", type=float, default=0.5, help="Reward for opening the door (once per episode)")
parser.add_argument("--key_drop_reward", type=float, default=0.0, help="Reward (usually negative) for dropping the key")
parser.add_argument("--device", type=str, default="cpu", help="Device to run on (cpu or cuda)")

# --- 1. Setup Phase ---

# # Mock function to simulate loading your checkpoints
# def load_my_policy(path): 
#     # Return your actual model object (e.g., a PyTorch nn.Module)
#     pass 

# # Initialize Registry
# registry = SkillRegistry()

# # Register skills from your different sources [cite: 87]
# # Example: 10 skills from DIAYN, 10 from DADS
# registry.register_baseline(load_my_policy, "../baseline_checkpoints/RSD/diayn_v1.pt", num_skills=8, z_dim=50)
# registry.register_baseline(load_my_policy, "../baseline_checkpoints/DADS/dads_v1.pt", num_skills=8, z_dim=10)

# # --- 2. Environment Setup ---

# # Initialize your MiniGrid environment
# # Ensure it is wrapped to provide flat observations or handle Dict observations if using MultiInputPolicy
# import minigrid
# base_env = gym.make("MiniGrid-Empty-8x8-v0", render_mode="rgb_array")
# base_env = minigrid.wrappers.ImgObsWrapper(base_env) # Example wrapper to get just the image

# # Wrap it with our Meta-Controller logic
# meta_env = MetaControllerEnv(base_env, registry, skill_duration=10)

def main(_A: argparse.Namespace):

    # initialize skill registry
    skill_registry = SkillRegistry(_A.skill_count_per_algo)

    # load model configs from config file
    config = load_config(_A.config_path)
    
    # load models via adapters (while feeding skill_registry to adapters)
    adapters = [load_model_from_config(m, skill_registry=skill_registry) for m in config.model_cfgs]
    model_interfaces = {adapter.algo_name: adapter for adapter in adapters}

    # initialize environment
    meta_env = MetaControllerEnv(skill_registry,
                                model_interfaces,
                                env_name=_A.env_name,
                                skill_duration=_A.skill_duration,
                                render_mode=_A.render_mode,
                                key_pickup_reward=_A.key_pickup_reward,
                                door_open_reward=_A.door_open_reward,
                                key_drop_reward=_A.key_drop_reward)

    # integrate later to shared environment init
    # env_factory, tmp_env = build_env_factory(_A.env_name)

    from stable_baselines3.common.env_checker import check_env
    # to make sure our meta environment is in the format stable baselines expects
    check_env(meta_env)

    # vectorize environment for PPO efficiency
    vec_env = DummyVecEnv([lambda: meta_env])

    # initialize model
    model = PPO(
        "MlpPolicy", # Use CNN if input is pixels (MiniGrid), "MlpPolicy" if flat
        vec_env,
        learning_rate=_A.learning_rate,
        n_steps=_A.n_steps,
        batch_size=_A.batch_size,
        gamma=_A.gamma,       
        verbose=_A.verbose,
        tensorboard_log=_A.tensorboard_log,
        device = _A.device
    )

    # train model
    print("Starting training of Meta-Controller...")

    checkpoint_callback = CheckpointCallback(
        save_freq=_A.checkpoint_freq * _A.n_steps,
        save_path=os.path.join(_A.save_path, "checkpoints"),
        name_prefix="rl_model",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    
    model.learn(total_timesteps=_A.num_timesteps, callback=checkpoint_callback)

    # save model
    model.save(_A.save_path)

if __name__ == "__main__":
    _A = parser.parse_args()
    main(_A)