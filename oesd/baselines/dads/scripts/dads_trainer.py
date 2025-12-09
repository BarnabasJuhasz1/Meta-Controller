import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class DADSTrainer:
    """
    Trainer class for the DADS algorithm.
    Encapsulates the PolicyNet, SkillDynamicsNet, and training logic.
    """

    def __init__(self, policy_net, skill_dynamics_net, value_net, cfg):
        self.policy_net = policy_net
        self.skill_dynamics_net = skill_dynamics_net
        self.value_net = value_net
        self.cfg = cfg

        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.policy_lr)
        self.dynamics_optimizer = optim.Adam(self.skill_dynamics_net.parameters(), lr=cfg.dynamics_lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=cfg.value_lr)

        # Replay buffer
        self.replay_buffer = []

    def add_to_buffer(self, transition):
        """Add a transition to the replay buffer."""
        self.replay_buffer.append(transition)
        if len(self.replay_buffer) > self.cfg.replay_size:
            self.replay_buffer.pop(0)

    def sample_from_buffer(self, batch_size):
        """Sample a batch of transitions from the replay buffer."""
        indices = torch.randint(0, len(self.replay_buffer), (batch_size,))
        batch = [self.replay_buffer[i] for i in indices]
        return batch

    def train_policy(self, batch):
        """Train the policy network using a batch of transitions."""
        states, skills, actions, next_states, rewards = zip(*batch)
        states = torch.stack([torch.tensor(s, device=self.cfg.device) for s in states])
        skills = torch.stack([torch.tensor(s, device=self.cfg.device) for s in skills])
        actions = torch.stack([torch.tensor(a, device=self.cfg.device) for a in actions])

        # Compute policy loss
        logits = self.policy_net(states, skills)
        loss = nn.CrossEntropyLoss()(logits, actions)

        # Optimize policy
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()

    def train_dynamics(self, batch):
        """Train the skill dynamics network using a batch of transitions."""
        states, skills, actions, next_states, rewards = zip(*batch)
        states = torch.stack([torch.as_tensor(s, device=self.cfg.device) for s in states])
        skills = torch.stack([torch.as_tensor(s, device=self.cfg.device) for s in skills])
        next_states = torch.stack([torch.as_tensor(ns, device=self.cfg.device).detach().clone() for ns in next_states])

        # Compute dynamics loss
        predicted_next_states, _ = self.skill_dynamics_net(states, skills)

        loss = nn.MSELoss()(predicted_next_states, next_states)

        # Optimize dynamics
        self.dynamics_optimizer.zero_grad()
        loss.backward()
        self.dynamics_optimizer.step()

    def train_value(self, batch):
        """Train the value network using a batch of transitions."""
        # Unpack only the relevant fields from the batch
        states, skills, _, _, rewards = zip(*batch)
        states = torch.stack([torch.tensor(s, device=self.cfg.device) for s in states])
        skills = torch.stack([torch.tensor(s, device=self.cfg.device) for s in skills])
        rewards = torch.stack([torch.tensor(r, device=self.cfg.device) for r in rewards])

        # Compute value loss
        predicted_values = self.value_net(states, skills)
        loss = nn.MSELoss()(predicted_values, rewards)

        # Optimize value network
        self.value_optimizer.zero_grad()
        loss.backward()
        self.value_optimizer.step()

    def train_step(self):
        """Perform a single training step."""
        if len(self.replay_buffer) < self.cfg.batch_size:
            return

        batch = self.sample_from_buffer(self.cfg.batch_size)
        self.train_policy(batch)
        self.train_dynamics(batch)
        self.train_value(batch)