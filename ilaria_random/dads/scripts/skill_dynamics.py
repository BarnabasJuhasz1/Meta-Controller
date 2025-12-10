# skill_dynamics_torch.py
#
# PyTorch implementation of the skill dynamics model used in DADS.
# Learns p(s_next | s, z) as a Gaussian over state deltas.

import torch
import torch.nn as nn
import torch.nn.functional as F


class SkillDynamics(nn.Module):
    """
    Dynamics model p(s_next | s, z).

    We model the distribution over the *delta*:
        delta = s_next - s

    as a multivariate diagonal Gaussian:
        delta ~ N(mu_delta(s, z), Sigma(s, z))

    Args:
        state_dim:   dimension of environment state s (without skill)
        skill_dim:   dimension of latent skill z
        hidden_dim:  hidden size of MLP
        fix_variance: if True, use unit variance; if False, learn std per dim
        std_min / std_max: clipping range for learned std
    """

    def __init__(
        self,
        state_dim,
        skill_dim,
        hidden_dim=256,
        fix_variance=True,
        std_min=0.3,
        std_max=10.0,
        device=None,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.skill_dim = skill_dim
        self.input_dim = state_dim + skill_dim
        self.hidden_dim = hidden_dim
        self.fix_variance = fix_variance
        self.std_min = std_min
        self.std_max = std_max

        self.device = device if device is not None else torch.device("cpu")

        # Simple MLP over [s, z]
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Mean of delta s
        self.mean_head = nn.Linear(hidden_dim, state_dim)

        # Either learn std or use fixed std = 1
        if not self.fix_variance:
            self.log_std_head = nn.Linear(hidden_dim, state_dim)
        else:
            self.log_std_head = None  # not used

        self.to(self.device)

    def forward(self, s, z):
        """
        Forward pass to get distribution parameters.

        Args:
            s: [B, state_dim]
            z: [B, skill_dim]

        Returns:
            mean_delta: [B, state_dim]
            log_std:    [B, state_dim]  (if fix_variance=False)
                        or tensor of zeros if fix_variance=True
        """
        x = torch.cat([s, z], dim=-1)  # [B, state_dim + skill_dim]
        h = self.net(x)
        mean_delta = self.mean_head(h)

        if self.fix_variance:
            # log_std = 0 => std = 1
            log_std = torch.zeros_like(mean_delta)
        else:
            log_std = self.log_std_head(h)
            log_std = torch.clamp(log_std, self.std_min, self.std_max)

        return mean_delta, log_std

    def log_prob(self, s_next, s, z):
        """
        Compute log p(s_next | s, z).

        Args:
            s_next: [B, state_dim]
            s:      [B, state_dim]
            z:      [B, skill_dim]

        Returns:
            logp: [B] tensor of log probabilities.
        """
        # delta will be modeled as Gaussian
        delta = s_next - s  # [B, state_dim]

        mean_delta, log_std = self.forward(s, z)  # both [B, state_dim]
        std = torch.exp(log_std)

        # Independent Normal across dimensions
        # log N(x; mu, sigma^2) = -0.5 * [ ((x-mu)/sigma)^2 + 2 log sigma + log(2π) ]
        var = std ** 2
        log_two_pi = torch.log(torch.tensor(2.0 * torch.pi, device=self.device))

        # [B, state_dim]
        log_prob_per_dim = -0.5 * (
            (delta - mean_delta) ** 2 / (var + 1e-8)
            + 2.0 * log_std
            + log_two_pi
        )
        # Sum over dims → [B]
        log_prob = log_prob_per_dim.sum(dim=-1)
        return log_prob

    def predict_state(self, s, z):
        """
        Predict the *mean* next state E[s_next | s, z].

        Args:
            s: [B, state_dim]
            z: [B, skill_dim]

        Returns:
            s_pred: [B, state_dim]
        """
        mean_delta, _ = self.forward(s, z)
        return s + mean_delta
