#!/usr/bin/env python3
"""
DADS-style skill discovery on MiniGrid DoorKey-8x8 in PyTorch.

Requirements (install in a Python 3.10+ env):

    pip install torch gymnasium minigrid numpy

Key design choices (so you can answer questions later):

1) Checkpoint loading
---------------------

We save checkpoints with:

    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "dynamics_state_dict": dynamics.state_dict(),
            "value_state_dict": value_net.state_dict(),
            "skills": skills_matrix,  # shape [num_skills, num_skills]
        },
        "dads_doorkey.pth",
    )

To load:

    import torch
    from dads_minigrid_doorkey import PolicyNet, SkillDynamicsNet, ValueNet

    num_skills = 8
    state_dim = 149        # see point (2) below
    action_dim = 7         # for MiniGrid DoorKey-8x8-v0

    ckpt = torch.load("dads_doorkey.pth", map_location="cpu")

    policy = PolicyNet(state_dim=state_dim, skill_dim=num_skills, action_dim=action_dim)
    policy.load_state_dict(ckpt["policy_state_dict"])

    skills = ckpt["skills"]  # tensor of shape [num_skills, num_skills]

2) Observation given to the agent
---------------------------------

We define a *base state* s as:

    - The 7×7×3 MiniGrid "image" (partial view) flattened and normalized to [0,1]
        -> 7 * 7 * 3 = 147 dims
    - A 4-dim one-hot encoding of the agent's direction (0..3)
        -> 4 dims (normalize to make it 1 dim)
    - A 1-dim float flag "has_key":
        has_key = 1.0 if env.carrying is not None else 0.0
        -> 1 dim

So:

    state_dim = 147 + 4 + 1 = 152

The policy actually sees:

    [ state (152 dims), skill_one_hot (num_skills dims) ]

No global (x, y) position is used. The agent only gets the local egocentric view
and the has_key flag.

"""

import math
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import gymnasium as gym
import minigrid
from oesd.baselines.dads.scripts.dads_trainer import DADSTrainer


# -------------------------- Config and utilities -------------------------- #

@dataclass
class Config:
    env_id: str = "MiniGrid-DoorKey-8x8-v0"
    num_skills: int = 8
    total_steps: int = 10_000
    warmup_steps: int = 5_000
    batch_size: int = 128
    dynamics_lr: float = 3e-4
    policy_lr: float = 3e-4
    value_lr: float = 3e-4
    gamma: float = 0.99

    # DADS reward parameters
    alt_skill_samples: int = 8   # number of alt skills for MI denominator

    # Network sizes
    hidden_dim: int = 256

    # Replay buffer
    replay_size: int = 200_000

    # Logging / checkpoint
    log_interval: int = 1_000
    ckpt_path: str = "dads_doorkey.pth"

    device: str = "cpu"


def make_env(env_id: str):
    """Create MiniGrid DoorKey env and return env, state_dim, action_dim."""
    env = gym.make(env_id)  # returns (obs, info) on reset

    # We assume obs["image"] is 7x7x3, obs["direction"] is int 0..3
    obs, info = env.reset()
    assert isinstance(obs, dict), "Expected dict observation from MiniGrid."

    image = obs["image"]
    h, w, c = image.shape  # should be 7,7,3
    image_dim = h * w * c  # 147

    direction_dim = 1  # Reduced from 4 to 1
    has_key_dim = 1

    state_dim = image_dim + direction_dim + has_key_dim
    #action_dim = env.action_space.n
    action_dim = 6 # MiniGrid DoorKey has 6 discrete actions

    return env, state_dim, action_dim


def extract_state(env, obs) -> np.ndarray:
    """
    Convert MiniGrid dict obs to a flat float32 vector:

    [flattened 7x7x3 image, 1-dim direction integer, 1-dim has_key flag]
    """
    image = obs["image"].astype(np.float32)  # shape (7,7,3)
    image_flat = (image / 10.0).reshape(-1)  # normalize a bit (colors are small ints)

    direction = np.array([obs["direction"]], dtype=np.float32)  # Use integer encoding

    # Use env internal state to define "has_key"
    has_key = 1.0 if getattr(env.unwrapped, "carrying", None) is not None else 0.0
    has_key_arr = np.array([has_key], dtype=np.float32)

    state = np.concatenate([image_flat, direction, has_key_arr], axis=0)
    return state


def sample_skill(num_skills: int, device: str):
    """Sample a discrete one-hot skill vector z."""
    k = random.randrange(num_skills)
    z = torch.zeros(num_skills, device=device)
    z[k] = 1.0
    return z, k


# -------------------------- Networks -------------------------- #

class PolicyNet(nn.Module):
    """Skill-conditioned policy for discrete actions."""

    def __init__(self, state_dim: int, skill_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + skill_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state, skill_onehot):
        x = torch.cat([state, skill_onehot], dim=-1)
        logits = self.net(x)
        return logits  # shape [B, action_dim]

    def act(self, state_np, skill_onehot, device="cpu"):
        """Sample an action given a single state np array and skill tensor."""
        state = torch.from_numpy(state_np).float().unsqueeze(0).to(device)  # [1, state_dim]
        skill = skill_onehot.unsqueeze(0).to(device)                        # [1, skill_dim]
        with torch.no_grad():
            logits = self.forward(state, skill)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs=probs)
            action = dist.sample()
        return int(action.item())


class SkillDynamicsNet(nn.Module):
    """
    p(s' | s, z) modeled as a diagonal Gaussian over the NEXT STATE (not delta).

    Input: concat(s, z_onehot)
    Output: mean, log_std over s'
    """

    def __init__(self, state_dim: int, skill_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.state_dim = state_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim + skill_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, state_dim)
        self.log_std_head = nn.Linear(hidden_dim, state_dim)

    def forward(self, state, skill_onehot):
        x = torch.cat([state, skill_onehot], dim=-1)
        h = self.net(x)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(-5.0, 2.0)  # reasonable bounds
        return mean, log_std

    def log_prob(self, next_state, state, skill_onehot):
        """
        Compute log p_\phi(s' | s, z) for a batch of transitions.
        Shapes:
            next_state: [B, state_dim]
            state:      [B, state_dim]
            skill:      [B, skill_dim]
        """
        mean, log_std = self.forward(state, skill_onehot)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        # Independent dims → sum of log probs
        logp = dist.log_prob(next_state).sum(dim=-1)  # [B]
        return logp


class ValueNet(nn.Module):
    """Baseline V(s,z) used for advantage estimation."""

    def __init__(self, state_dim: int, skill_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + skill_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, skill_onehot):
        x = torch.cat([state, skill_onehot], dim=-1)
        v = self.net(x)
        return v.squeeze(-1)  # [B]


# -------------------------- Replay buffer -------------------------- #

class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, num_skills: int):
        self.capacity = capacity
        self.state_dim = state_dim
        self.num_skills = num_skills

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.skills = np.zeros((capacity, num_skills), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

        self.idx = 0
        self.full = False

    def add(self, s, z_onehot, a, s_next, done):
        self.states[self.idx] = s
        self.skills[self.idx] = z_onehot.cpu().numpy()
        self.actions[self.idx] = a
        self.next_states[self.idx] = s_next
        self.dones[self.idx] = float(done)

        self.idx = (self.idx + 1) % self.capacity
        if self.idx == 0:
            self.full = True

    def size(self):
        return self.capacity if self.full else self.idx

    def sample(self, batch_size: int, device: str):
        n = self.size()
        idxs = np.random.randint(0, n, size=batch_size)
        batch = dict(
            states=torch.from_numpy(self.states[idxs]).to(device),
            next_states=torch.from_numpy(self.next_states[idxs]).to(device),
            actions=torch.from_numpy(self.actions[idxs]).to(device),
            skills=torch.from_numpy(self.skills[idxs]).to(device),
            dones=torch.from_numpy(self.dones[idxs]).to(device),
        )
        return batch


# -------------------------- DADS reward -------------------------- #

def compute_dads_reward(
    dynamics: SkillDynamicsNet,
    states: torch.Tensor,
    skills: torch.Tensor,
    next_states: torch.Tensor,
    num_skills: int,
    alt_samples: int,
) -> torch.Tensor:
    """
    Compute intrinsic reward r(s,z,s') using DADS MI estimator:

        r = log(K) - log( sum_{z'} exp(log p(s'|s,z') - log p(s'|s,z)) )

    approximated by K' = (alt_samples + 1) different skills (the given one + sampled).

    Inputs shapes: [B, ...]. Returns tensor [B].
    """
    device = states.device
    B = states.shape[0]

    # log p(s'|s,z)
    logp = dynamics.log_prob(next_states, states, skills)  # [B]

    # Build alt skills: include the current skill plus alt_samples sampled skills
    # for each sample in the batch.
    with torch.no_grad():
        # shape [B, alt_samples, num_skills]
        alt_skills_list = []
        for _ in range(alt_samples):
            # Sample skills uniformly
            k = torch.randint(0, num_skills, (B,), device=device)
            one_hot = torch.zeros(B, num_skills, device=device)
            one_hot[torch.arange(B), k] = 1.0
            alt_skills_list.append(one_hot)
        alt_skills = torch.stack(alt_skills_list, dim=0)  # [alt_samples, B, K]

    # Repeat states / next_states for alt skills
    states_rep = states.unsqueeze(0).expand(alt_samples, B, -1)       # [alt_samples,B,D]
    next_rep = next_states.unsqueeze(0).expand(alt_samples, B, -1)   # [alt_samples,B,D]

    # Flatten for batch processing
    states_flat = states_rep.reshape(-1, states.shape[-1])
    next_flat = next_rep.reshape(-1, next_states.shape[-1])
    skills_flat = alt_skills.reshape(-1, skills.shape[-1])

    logp_alt_flat = dynamics.log_prob(next_flat, states_flat, skills_flat)  # [alt_samples*B]
    logp_alt = logp_alt_flat.view(alt_samples, B)                            # [alt_samples,B]

    # DADS formula (log-sum-exp style)
    # r = log(N+1) - log(1 + sum_{i} exp(logp_alt[i] - logp))
    # broadcast logp: [1,B]
    diff = logp_alt - logp.unsqueeze(0)  # [alt_samples,B]
    # for numerical stability, clamp diff
    diff = diff.clamp(-50.0, 50.0)
    denom = 1.0 + torch.exp(diff).sum(dim=0)  # [B]
    r = math.log(alt_samples + 1.0) - torch.log(denom)
    return r.detach()


# -------------------------- Training loop -------------------------- #

def train_dads(cfg: Config):
    device = cfg.device
    print(f"Using device: {device}")

    env, state_dim, action_dim = make_env(cfg.env_id)
    print(f"Env: {cfg.env_id}, state_dim={state_dim}, action_dim={action_dim}")

    policy = PolicyNet(state_dim, cfg.num_skills, action_dim, cfg.hidden_dim).to(device)
    dynamics = SkillDynamicsNet(state_dim, cfg.num_skills, cfg.hidden_dim).to(device)
    value_net = ValueNet(state_dim, cfg.num_skills, cfg.hidden_dim).to(device)

    trainer = DADSTrainer(policy, dynamics, value_net, cfg)

    # Initialize episode
    obs, info = env.reset()
    state = extract_state(env, obs)
    skill_onehot, skill_idx = sample_skill(cfg.num_skills, device)

    for step in range(1, cfg.total_steps + 1):
        # Act
        action = policy.act(state, skill_onehot, device=device)
        next_obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        next_state = extract_state(env, next_obs)

        # Store transition
        trainer.add_to_buffer((state, skill_onehot, action, next_state, 0.0))

        # Train
        trainer.train_step()

        state = next_state

        # Log progress every 100 steps
        if step % 100 == 0:
            print(f"Step {step}/{cfg.total_steps} completed.")

    env.close()
    print("Training finished. Checkpoint saved to:", cfg.ckpt_path)


# -------------------------- Main entry ------------------------------------ #

if __name__ == "__main__":
    cfg = Config()
    train_dads(cfg)
