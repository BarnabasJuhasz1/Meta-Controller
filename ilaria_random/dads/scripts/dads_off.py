# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import, division, print_function

import os
import sys
import time
import io
import pickle as pkl
import functools

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch

from absl import flags, logging, app

sys.path.append(os.path.abspath("./"))

import gymnasium as gym
from minigrid.wrappers import FullyObsWrapper

from envs import skill_wrapper
#from envs import video_wrapper  # if you have a Gym wrapper for video

# This will be the PyTorch port of dads_agent.py
from dads_agent import DADSAgent  # <- we'll rewrite this next

FLAGS = flags.FLAGS

# ---------------------------------------------------------------------
# Flags (mostly kept the same as original for compatibility)
# ---------------------------------------------------------------------

flags.DEFINE_string("logdir", "~/tmp/dads", "Directory for saving experiment data")

# environment hyperparameters
flags.DEFINE_string(
    "environment",
    "MiniGrid-Empty-8x8-v0",
    "Name of the environment (MiniGrid ID)",
)
flags.DEFINE_integer("max_env_steps", 200, "Maximum number of steps in one episode")
flags.DEFINE_integer(
    "reduced_observation",
    0,
    "Predict dynamics in a reduced observation space (0 = use raw obs)",
)
flags.DEFINE_integer(
    "min_steps_before_resample",
    50,
    "Minimum number of steps to execute before resampling skill",
)
flags.DEFINE_float(
    "resample_prob",
    0.0,
    "Creates stochasticity timesteps before resampling skill",
)

# saving / recording
flags.DEFINE_string(
    "save_model",
    None,
    "Name to save the model with, None implies the models are not saved.",
)
flags.DEFINE_integer("save_freq", 100, "Saving frequency (in epochs)")
flags.DEFINE_string(
    "vid_name",
    None,
    "Base name for videos being saved, None implies videos are not recorded",
)
flags.DEFINE_integer("record_freq", 100, "Video recording frequency (in epochs)")

# final evaluation after training is done
flags.DEFINE_integer("run_eval", 0, "Evaluate learnt skills at the end")

# evaluation type
flags.DEFINE_integer("num_evals", 0, "Number of skills to evaluate")
flags.DEFINE_integer(
    "deterministic_eval",
    0,
    "Evaluate all skills deterministically (only works for discrete skills)",
)

# training
flags.DEFINE_integer("run_train", 0, "Train the agent")
flags.DEFINE_integer("num_epochs", 500, "Number of training epochs")

# skill latent space
flags.DEFINE_integer("num_skills", 2, "Number of skills to learn")
flags.DEFINE_string(
    "skill_type",
    "cont_uniform",
    "Type of skill and the prior over it (only cont_uniform used here)",
)

# network size hyperparameter
flags.DEFINE_integer(
    "hidden_layer_size",
    512,
    "Hidden layer size, shared by actor, critic and dynamics",
)

# reward structure
flags.DEFINE_integer(
    "random_skills",
    0,
    "Number of skills to sample randomly for approximating mutual information",
)

# optimization hyperparameters
flags.DEFINE_integer(
    "replay_buffer_capacity",
    int(1e6),
    "Capacity of the replay buffer",
)
flags.DEFINE_integer(
    "initial_collect_steps",
    2000,
    "Steps collected initially before training to populate the buffer",
)
flags.DEFINE_integer("collect_steps", 200, "Steps collected per epoch")
flags.DEFINE_integer(
    "agent_batch_size",
    256,
    "Batch size for agent updates",
)
flags.DEFINE_integer(
    "agent_train_steps",
    128,
    "Number of gradient steps per epoch",
)
flags.DEFINE_float("agent_lr", 3e-4, "Learning rate for the agent")
flags.DEFINE_float(
    "skill_dynamics_lr",
    3e-4,
    "Learning rate for the skill dynamics model",
)

# SAC-ish hyperparameters
flags.DEFINE_float("agent_entropy", 0.1, "Entropy regularization coefficient")
flags.DEFINE_float("agent_gamma", 0.99, "Reward discount factor")
flags.DEFINE_float(
    "action_clipping",
    1.0,
    "Clip actions to (-eps, eps) per dimension to avoid tanh issues",
)

flags.DEFINE_integer("debug", 0, "Extra debug prints")
flags.DEFINE_integer(
    "device",
    0,
    "Device index (ignored if CUDA not available, CPU used otherwise)",
)

# ---------------------------------------------------------------------
# Simple replay buffer (PyTorch-compatible)
# ---------------------------------------------------------------------


class ReplayBuffer(object):
    """Very simple replay buffer for off-policy RL."""

    def __init__(self, capacity, obs_dim, act_dim):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.ptr = 0
        self.size = 0

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            obs=self.obs[idxs],
            actions=self.actions[idxs],
            rewards=self.rewards[idxs],
            next_obs=self.next_obs[idxs],
            dones=self.dones[idxs],
        )
        return batch


# ---------------------------------------------------------------------
# Environment utilities
# ---------------------------------------------------------------------


def get_environment(env_name=None, max_episode_steps=None):
    """Create a MiniGrid environment (Gymnasium) and optionally wrap it."""
    if env_name is None:
        env_name = FLAGS.environment

    if "MiniGrid" not in env_name:
        raise ValueError(
            f"This script is configured only for MiniGrid envs, got env_name={env_name}"
        )

    env = gym.make(env_name)
    # If you want full matrix obs, you could uncomment this:
    # env = FullyObsWrapper(env)

    # Optionally enforce max episode length at Gymnasium level
    if max_episode_steps is not None:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)

    return env


def process_observation(observation):
    """Optionally reduce MiniGrid observations.

    Right now, if reduced_observation == 0, we just flatten the observation.
    You can customize this to match whatever you use in other baselines.
    """

    if FLAGS.reduced_observation == 0:
        # Default MiniGrid obs from FullyObsWrapper is a dict with 'image'
        # Otherwise, it can be a 3D array.
        if isinstance(observation, dict) and "image" in observation:
            obs = observation["image"]
        else:
            obs = observation
        return np.asarray(obs, dtype=np.float32).flatten()

    # Example low-dimensional branch (if you use custom wrappers)
    if isinstance(observation, dict) and "agent_pos" in observation:
        x, y = observation["agent_pos"]
        direction = observation["agent_dir"]  # 0,1,2,3

        dir_onehot = np.zeros(4, dtype=np.float32)
        dir_onehot[direction] = 1.0

        red_obs = np.array([x, y], dtype=np.float32)
        red_obs = np.concatenate([red_obs, dir_onehot], axis=0)
        return red_obs

    return np.asarray(observation, dtype=np.float32).flatten()


# ---------------------------------------------------------------------
# Training / evaluation loops
# ---------------------------------------------------------------------


def collect_experience(env, agent, buffer, num_steps, skill_dim):
    """Collect transitions using the current agent policy."""
    episode_returns = []
    episode_lengths = []

    obs, info = env.reset()
    obs_vec = process_observation(obs)

    ep_ret = 0.0
    ep_len = 0

    for _ in range(num_steps):
        # Agent expects full observation including skill in obs_vec.
        # Here we assume skill_wrapper already appended z to obs, so
        # obs_vec already has shape [state_dim + skill_dim].
        action = agent.select_action(obs_vec, deterministic=False)
        if FLAGS.action_clipping < 1.0:
            action = np.clip(action, -FLAGS.action_clipping, FLAGS.action_clipping)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        next_obs_vec = process_observation(next_obs)

        buffer.add(obs_vec, action, reward, next_obs_vec, done)

        ep_ret += reward
        ep_len += 1

        obs_vec = next_obs_vec

        if done:
            episode_returns.append(ep_ret)
            episode_lengths.append(ep_len)
            obs, info = env.reset()
            obs_vec = process_observation(obs)
            ep_ret = 0.0
            ep_len = 0

    return episode_returns, episode_lengths


def evaluate(env, agent, num_episodes, skill_dim):
    """Run evaluation episodes with fixed (or random) skills."""
    returns = []

    for _ in range(num_episodes):
        obs, info = env.reset()
        obs_vec = process_observation(obs)
        ep_ret = 0.0
        done = False

        while not done:
            action = agent.select_action(obs_vec, deterministic=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            next_obs_vec = process_observation(next_obs)
            ep_ret += reward
            obs_vec = next_obs_vec

        returns.append(ep_ret)

    return np.mean(returns), returns


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main(_):
    logging.set_verbosity(logging.INFO)

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)

    # directories
    root_dir = os.path.abspath(os.path.expanduser(FLAGS.logdir))
    os.makedirs(root_dir, exist_ok=True)
    log_dir = os.path.join(root_dir, FLAGS.environment)
    os.makedirs(log_dir, exist_ok=True)
    save_dir = os.path.join(log_dir, "models")
    os.makedirs(save_dir, exist_ok=True)

    print("directory for recording experiment data:", log_dir)

    # metric buffers (reloaded if continuing training)
    global sample_count, iter_count, episode_size_buffer, episode_return_buffer
    try:
        sample_count = np.load(os.path.join(log_dir, "sample_count.npy")).tolist()
        iter_count = np.load(os.path.join(log_dir, "iter_count.npy")).tolist()
        episode_size_buffer = np.load(
            os.path.join(log_dir, "episode_size_buffer.npy")
        ).tolist()
        episode_return_buffer = np.load(
            os.path.join(log_dir, "episode_return_buffer.npy")
        ).tolist()
    except Exception:
        sample_count = 0
        iter_count = 0
        episode_size_buffer = []
        episode_return_buffer = []

    # ------------------------------------------------------------------
    # Environment and agent setup
    # ------------------------------------------------------------------
    base_env = get_environment(env_name=FLAGS.environment, max_episode_steps=FLAGS.max_env_steps)

    # Wrap with skill wrapper: adds latent skill z to observation, resamples every K steps
    env = skill_wrapper.SkillWrapper(
        base_env,
        num_latent_skills=FLAGS.num_skills,
        skill_type=FLAGS.skill_type,
        preset_skill=None,
        min_steps_before_resample=FLAGS.min_steps_before_resample,
        resample_prob=FLAGS.resample_prob,
    )

    # Build obs/action dimensions
    # We assume env.observation_space is Box and env.action_space is Box
    # (for MiniGrid + continuous control wrappers).
    dummy_obs, _ = env.reset()
    obs_vec = process_observation(dummy_obs)
    obs_dim = obs_vec.shape[0]

    if hasattr(env.action_space, "shape"):
        act_dim = int(np.prod(env.action_space.shape))
    else:
        # If Discrete, you will need a different policy; for now, assume Box
        raise ValueError("Current DADS script assumes continuous (Box) actions.")

    skill_dim = FLAGS.num_skills

    # Replay buffer
    replay_buffer = ReplayBuffer(
        capacity=FLAGS.replay_buffer_capacity,
        obs_dim=obs_dim,
        act_dim=act_dim,
    )

    # Agent (PyTorch DADS)
    agent = DADSAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        skill_dim=skill_dim,
        hidden_dim=FLAGS.hidden_layer_size,
        gamma=FLAGS.agent_gamma,
        entropy_coef=FLAGS.agent_entropy,
        lr_actor=FLAGS.agent_lr,
        lr_critic=FLAGS.agent_lr,
        lr_dynamics=FLAGS.skill_dynamics_lr,
        device=device,
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    start_time = time.time()

    if FLAGS.run_train:
        # Initial random data collection
        if iter_count == 0 and replay_buffer.size < FLAGS.initial_collect_steps:
            print("Collecting initial random experience...")
            # Use a random policy at the beginning
            class RandomPolicy:
                def __init__(self, act_space):
                    self.act_space = act_space

                def select_action(self, obs, deterministic=False):
                    return self.act_space.sample()

            random_agent = RandomPolicy(env.action_space)
            ep_rets, ep_lens = collect_experience(
                env, random_agent, replay_buffer, FLAGS.initial_collect_steps, skill_dim
            )
            sample_count += FLAGS.initial_collect_steps
            episode_return_buffer.extend(ep_rets)
            episode_size_buffer.extend(ep_lens)
            print("Initial collection done.")

        # Main training epochs
        while iter_count < FLAGS.num_epochs:
            print("Epoch:", iter_count)

            # 1) Collect experience using current policy
            ep_rets, ep_lens = collect_experience(
                env, agent, replay_buffer, FLAGS.collect_steps, skill_dim
            )
            sample_count += FLAGS.collect_steps
            if len(ep_rets) > 0:
                episode_return_buffer.extend(ep_rets)
                episode_size_buffer.extend(ep_lens)

            # 2) Train agent for agent_train_steps gradient steps
            if replay_buffer.size >= FLAGS.agent_batch_size:
                for _ in range(FLAGS.agent_train_steps):
                    batch = replay_buffer.sample(FLAGS.agent_batch_size)
                    info = agent.update(batch)

                if FLAGS.debug:
                    print(
                        "losses:",
                        {k: float(v) for k, v in info.items()},
                    )

            # 3) Save models & metrics
            if FLAGS.save_model is not None and (iter_count % FLAGS.save_freq == 0):
                print("Saving models at epoch", iter_count)
                agent.save(save_dir, prefix=f"{FLAGS.save_model}_epoch{iter_count}")
                np.save(os.path.join(log_dir, "sample_count.npy"), sample_count)
                np.save(os.path.join(log_dir, "episode_size_buffer.npy"), episode_size_buffer)
                np.save(os.path.join(log_dir, "episode_return_buffer.npy"), episode_return_buffer)
                np.save(os.path.join(log_dir, "iter_count.npy"), iter_count)

            # 4) Periodic evaluation (without videos for now)
            if FLAGS.record_freq is not None and (iter_count % FLAGS.record_freq == 0):
                avg_ret, _ = evaluate(env, agent, num_episodes=5, skill_dim=skill_dim)
                print(f"[Eval] Epoch {iter_count} - avg return: {avg_ret:.3f}")

            iter_count += 1

        print("Training finished in %.2f seconds." % (time.time() - start_time))

    # ------------------------------------------------------------------
    # Final evaluation (optional)
    # ------------------------------------------------------------------
    if FLAGS.run_eval:
        avg_ret, all_rets = evaluate(env, agent, num_episodes=FLAGS.num_evals or 10, skill_dim=skill_dim)
        print("Final evaluation average return:", avg_ret)
        with open(os.path.join(log_dir, "final_eval_returns.pkl"), "wb") as f:
            pkl.dump(all_rets, f)


if __name__ == "__main__":
    app.run(main)
