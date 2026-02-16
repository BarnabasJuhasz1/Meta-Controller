
# pip install gymnasium minigrid gym-minigrid

import sys, os
sys.path.append(os.path.dirname(__file__))

import gymnasium as gym
# from SimpleEnv import example_minigrid
from example_minigrid import SimpleEnv


def random_sample_policy(obs):
    return env.action_space.sample()

def train(env, observation, policy):

   for _ in range(1000):
      action = policy(observation)  # User-defined policy function
      observation, reward, terminated, truncated, info = env.step(action)

      if terminated or truncated:
         observation, info = env.reset()

   env.close()

   
if __name__ == "__main__":

   # print(gym.envs.registry.keys())
   # env = gym.make("MiniGrid-Empty-5x5-v0", render_mode="human")
   env = SimpleEnv(render_mode="human")

   observation, info = env.reset(seed=42) 

   train(env, observation, random_sample_policy)