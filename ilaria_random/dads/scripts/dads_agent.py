# dads_agent_torch.py
#
# PyTorch implementation of the DADS agent.
# Replaces the original TensorFlow/TF-Agents DADSAgent.

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from skill_dynamics import SkillDynamics  # we'll port this next


class GaussianPolicy(nn.Module):
    """
    Gaussian policy with Tanh squashing (SAC-style) for continuous actions.
    Outputs actions in [-1, 1]^act_dim.
    """

    def __init__(self, obs_dim, act_dim, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, act_dim)
        self.log_std_head = nn.Linear(hidden_dim, act_dim)

    def forward(self, obs):
        """
        obs: [B, obs_dim]
        returns: mean [B, act_dim], log_std [B, act_dim]
        """
        h = self.net(obs)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, obs):
        """
        obs: [B, obs_dim]
        returns: action [B, act_dim], log_prob [B], mean_action [B, act_dim]
        """
        mean, log_std = self.forward(obs)
        std = log_std.exp()

        # Reparameterization trick
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()  # [B, act_dim]
        action = torch.tanh(z)  # [-1,1]

        # Compute log prob of action under Tanh Gaussian
        log_prob = normal.log_prob(z).sum(dim=-1)
        # log probability adjustment for Tanh (from SAC)
        log_prob -= torch.log(1 - action.pow(2) + 1e-7).sum(dim=-1)

        mean_action = torch.tanh(mean)
        return action, log_prob, mean_action


class QNetwork(nn.Module):
    """
    Q(s, a) network. Takes concatenated [obs, action] as input.
    """

    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs, action):
        """
        obs:    [B, obs_dim]
        action: [B, act_dim]
        returns: Q value [B, 1]
        """
        x = torch.cat([obs, action], dim=-1)
        q = self.net(x)
        return q


class DADSAgent:
    """
    PyTorch DADS agent.

    - SAC-style actor & critic
    - Skill dynamics model p_phi(s'|s,z)
    - Intrinsic reward: r_DADS(s,s',z) ~ I(s'; z | s)
    """

    def __init__(
        self,
        obs_dim,
        act_dim,
        skill_dim,
        hidden_dim=256,
        gamma=0.99,
        entropy_coef=0.1,
        lr_actor=3e-4,
        lr_critic=3e-4,
        lr_dynamics=3e-4,
        device=None,
        latent_prior="cont_uniform",
        prior_samples=100,
    ):
        """
        obs_dim:    dimension of full observation (including skill z)
        act_dim:    dimension of action
        skill_dim:  dimension of latent skill z
        hidden_dim: hidden size for networks
        gamma:      discount factor
        entropy_coef: fixed alpha for SAC
        lr_*:       learning rates
        device:     torch.device
        latent_prior: 'cont_uniform' etc. (we implement 'cont_uniform' here)
        prior_samples: K, number of alt skills for denominator approximation
        """
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.skill_dim = skill_dim
        self.state_dim = obs_dim - skill_dim
        assert self.state_dim > 0, "obs_dim must be > skill_dim"

        self.gamma = gamma
        self.alpha = entropy_coef
        self.latent_prior = latent_prior
        self.prior_samples = prior_samples
        self.device = device if device is not None else torch.device("cpu")

        # Actor & critic
        self.actor = GaussianPolicy(obs_dim, act_dim, hidden_dim=hidden_dim).to(self.device)
        self.q_net = QNetwork(obs_dim, act_dim, hidden_dim=hidden_dim).to(self.device)
        self.q_target = QNetwork(obs_dim, act_dim, hidden_dim=hidden_dim).to(self.device)
        self.q_target.load_state_dict(self.q_net.state_dict())
        self.q_target.eval()

        # Skill dynamics model
        self.dynamics = SkillDynamics(
            state_dim=self.state_dim,
            skill_dim=self.skill_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)

        # Optimizers
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.q_opt = optim.Adam(self.q_net.parameters(), lr=lr_critic)
        self.dyn_opt = optim.Adam(self.dynamics.parameters(), lr=lr_dynamics)

        self.target_tau = 0.005  # soft target update

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------

    def select_action(self, obs_vec, deterministic=False):
        """
        obs_vec: numpy array [obs_dim]
        returns: numpy array [act_dim] in [-1,1]
        """
        self.actor.eval()
        with torch.no_grad():
            obs = torch.as_tensor(obs_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
            if deterministic:
                mean, _ = self.actor.forward(obs)
                action = torch.tanh(mean)
            else:
                action, _, _ = self.actor.sample(obs)
        self.actor.train()
        return action.cpu().numpy()[0]

    # ------------------------------------------------------------------
    # DADS intrinsic reward
    # ------------------------------------------------------------------

    def _sample_alt_skills(self, batch_size):
        """
        Sample alt skills from the prior.
        Returns: z_alt [B*K, skill_dim]
        """
        K = self.prior_samples if self.prior_samples > 0 else self.skill_dim - 1
        if self.latent_prior == "cont_uniform":
            z_alt = torch.empty(batch_size * K, self.skill_dim, device=self.device).uniform_(-1.0, 1.0)
        else:
            # For now, just support cont_uniform; others can be added later.
            z_alt = torch.empty(batch_size * K, self.skill_dim, device=self.device).uniform_(-1.0, 1.0)
        return z_alt, K

    def _compute_dads_reward_torch(self, s, z, s_next):
        """
        Torch version of compute_dads_reward.

        s:      [B, state_dim]
        z:      [B, skill_dim]
        s_next: [B, state_dim]

        Returns:
            r_dads: [B]
            info:   dict with logp, logp_altz (both detached to CPU)
        """
        B = s.shape[0]

        # log p(s_next | s, z)
        logp = self.dynamics.log_prob(s_next, s, z)  # [B]

        # sample alt skills for denominator
        z_alt, K = self._sample_alt_skills(B)

        # repeat s, s_next K times
        s_rep = s.unsqueeze(1).expand(-1, K, -1).reshape(-1, self.state_dim)         # [B*K, state_dim]
        s_next_rep = s_next.unsqueeze(1).expand(-1, K, -1).reshape(-1, self.state_dim)  # [B*K, state_dim]

        logp_altz_flat = self.dynamics.log_prob(s_next_rep, s_rep, z_alt)  # [B*K]
        logp_altz = logp_altz_flat.view(B, K)  # [B, K]

        # Intrinsic reward (same formula as original, but vectorized):
        # r = log(K+1) - log(1 + sum_z' exp(logp_altz - logp))
        # where logp is [B], broadcast to [B,K]
        logp_expanded = logp.unsqueeze(1).expand_as(logp_altz)  # [B, K]
        diff = torch.clamp(logp_altz - logp_expanded, -50.0, 50.0)
        denom = 1.0 + torch.exp(diff).sum(dim=1)  # [B]
        r_dads = torch.log(torch.tensor(float(K + 1), device=self.device)) - torch.log(denom + 1e-8)

        info = {
            "logp": logp.detach().cpu().numpy(),
            "logp_altz": logp_altz_flat.detach().cpu().numpy(),
        }
        return r_dads, info

    # ------------------------------------------------------------------
    # Update (RL + dynamics)
    # ------------------------------------------------------------------

    def update(self, batch):
        """
        Perform one update step of:
            - skill dynamics
            - critic Q
            - policy actor

        batch: dict with numpy arrays:
            - 'obs':      [B, obs_dim]
            - 'actions':  [B, act_dim]
            - 'rewards':  [B]        (unused, we use intrinsic reward instead)
            - 'next_obs': [B, obs_dim]
            - 'dones':    [B]
        """
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device)

        # Split obs into (state, skill)
        s = obs[:, : self.state_dim]                    # [B, state_dim]
        z = obs[:, self.state_dim :]                    # [B, skill_dim]
        s_next = next_obs[:, : self.state_dim]          # [B, state_dim]

        # 1) Train skill dynamics to maximize log p(s_next | s, z)
        logp_dyn = self.dynamics.log_prob(s_next, s, z)   # [B]
        dyn_loss = -logp_dyn.mean()

        self.dyn_opt.zero_grad()
        dyn_loss.backward()
        self.dyn_opt.step()

        # 2) Compute DADS intrinsic reward
        with torch.no_grad():
            r_dads, dads_info = self._compute_dads_reward_torch(s, z, s_next)

        # 3) SAC-style critic update using intrinsic reward only
        #    target Q: r + gamma * (1-done) * (Q_target(s', a') - alpha * log_pi)
        with torch.no_grad():
            next_action, next_logp, _ = self.actor.sample(next_obs)
            target_q = self.q_target(next_obs, next_action).squeeze(-1)  # [B]
            target_v = target_q - self.alpha * next_logp  # [B]
            y = r_dads + self.gamma * (1.0 - dones) * target_v  # [B]

        q = self.q_net(obs, actions).squeeze(-1)  # [B]
        q_loss = F.mse_loss(q, y)

        self.q_opt.zero_grad()
        q_loss.backward()
        self.q_opt.step()

        # 4) Actor update (maximize Q - alpha * log_pi)
        action_pi, logp_pi, _ = self.actor.sample(obs)
        q_pi = self.q_net(obs, action_pi).squeeze(-1)
        actor_loss = (self.alpha * logp_pi - q_pi).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # 5) Soft target update
        with torch.no_grad():
            for param, target_param in zip(self.q_net.parameters(), self.q_target.parameters()):
                target_param.data.mul_(1.0 - self.target_tau)
                target_param.data.add_(self.target_tau * param.data)

        info = {
            "dyn_loss": float(dyn_loss.item()),
            "critic_loss": float(q_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "logp_mean": float(dads_info["logp"].mean()),
            "r_dads_mean": float(r_dads.mean().item()),
        }
        return info

    # ------------------------------------------------------------------
    # Saving / loading
    # ------------------------------------------------------------------

    def save(self, save_dir, prefix="dads"):
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{prefix}.pt")
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "q_net": self.q_net.state_dict(),
                "q_target": self.q_target.state_dict(),
                "dynamics": self.dynamics.state_dict(),
                "actor_opt": self.actor_opt.state_dict(),
                "q_opt": self.q_opt.state_dict(),
                "dyn_opt": self.dyn_opt.state_dict(),
            },
            path,
        )

    def load(self, save_dir, prefix="dads"):
        path = os.path.join(save_dir, f"{prefix}.pt")
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.q_net.load_state_dict(ckpt["q_net"])
        self.q_target.load_state_dict(ckpt["q_target"])
        self.dynamics.load_state_dict(ckpt["dynamics"])
        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.q_opt.load_state_dict(ckpt["q_opt"])
        self.dyn_opt.load_state_dict(ckpt["dyn_opt"])

    # ------------------------------------------------------------------
    # Skill dynamics accessor
    # ------------------------------------------------------------------

    @property
    def skill_dynamics(self):
        return self.dynamics
