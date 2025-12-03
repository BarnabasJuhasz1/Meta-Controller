
import gymnasium as gym
import minigrid # <--- REQUIRED IMPORT
from wrappers import HybridWrapper

def make_env(env_id):
    def thunk():
        env = gym.make(env_id, render_mode="rgb_array")
        env = HybridWrapper(env)
        return env
    return thunk
