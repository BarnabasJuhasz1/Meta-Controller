# Algos/LSD/policy.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import numpy as np

from Interface.policy_base import SkillPolicy


# ============================================================
# φ-NETWORK (1-Lipschitz)
# ============================================================

class PhiNet(nn.Module):
    """
    1-Lipschitz φ(s) network using Spectral Norm.
    φ : obs_dim → skill_dim
    """
    def __init__(self, obs_dim, skill_dim, hidden_dim):
        super().__init__()

        self.net = nn.Sequential(
            spectral_norm(nn.Linear(obs_dim, hidden_dim)),
            nn.ReLU(),
            spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            spectral_norm(nn.Linear(hidden_dim, skill_dim)),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# POLICY + VALUE NETWORK (Actor-Critic)
# ============================================================

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, skill_dim, action_dim, hidden_dim):
        super().__init__()
        inp = obs_dim + skill_dim

        self.trunk = nn.Sequential(
            nn.Linear(inp, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs, z):
        h = self.trunk(torch.cat([obs, z], dim=-1))
        return self.policy_head(h), self.value_head(h)


# ============================================================
# LSD POLICY WRAPPER (unified interface)
# ============================================================

class LSDPolicy(SkillPolicy):
    """
    Wraps actor + φ-net for inference.
    """

    def __init__(self, actor, phi_net, num_skills, device="cpu"):
        self.actor = actor
        self.phi_net = phi_net
        self.num_skills = num_skills
        self.device = device

    def act(self, obs, skill):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        z_t = torch.tensor(skill, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            logits, _ = self.actor(obs_t, z_t)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()

        return action.item()

    def phi(self, obs):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            phi_vec = self.phi_net(obs_t)
        return phi_vec.cpu().numpy()[0]

    def skill_dim(self):
        return self.num_skills

    def skill_type(self):
        return "discrete"
