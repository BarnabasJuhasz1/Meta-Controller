from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback

from skill_registry import SkillRegistry
from meta_env_wrapper import MetaControllerEnv

# --- 1. Setup Phase ---

# Mock function to simulate loading your checkpoints
def load_my_policy(path): 
    # Return your actual model object (e.g., a PyTorch nn.Module)
    pass 

# Initialize Registry
registry = SkillRegistry()

# Register skills from your different sources [cite: 87]
# Example: 10 skills from DIAYN, 10 from DADS
registry.register_baseline(load_my_policy, "../baseline_checkpoints/RSD/diayn_v1.pt", num_skills=8, z_dim=50)
registry.register_baseline(load_my_policy, "../baseline_checkpoints/DADS/dads_v1.pt", num_skills=8, z_dim=10)

# --- 2. Environment Setup ---

# Initialize your MiniGrid environment
# Ensure it is wrapped to provide flat observations or handle Dict observations if using MultiInputPolicy
import minigrid
base_env = gym.make("MiniGrid-Empty-8x8-v0", render_mode="rgb_array")
base_env = minigrid.wrappers.ImgObsWrapper(base_env) # Example wrapper to get just the image

# Wrap it with our Meta-Controller logic
meta_env = MetaControllerEnv(base_env, registry, skill_duration=10)

# Vectorize for PPO efficiency
vec_env = DummyVecEnv([lambda: meta_env])

# --- 3. The Meta-Controller Model ---

# We use PPO as the high-level policy learner [cite: 10, 11]
model = PPO(
    "CnnPolicy",       # Use CNN if input is pixels (MiniGrid), "MlpPolicy" if flat
    vec_env,
    learning_rate=3e-4,
    n_steps=2048,      # Experience buffer size
    batch_size=64,
    gamma=0.99,        # Discount factor
    verbose=1,
    tensorboard_log="./meta_ppo_tensorboard/"
)

# --- 4. Train ---
print("Starting training of Meta-Controller...")
# This will train the manager to select the best skill index for the task
model.learn(total_timesteps=100000)

# --- 5. Save ---
model.save("meta_controller_v1")