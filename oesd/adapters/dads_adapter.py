from __future__ import annotations
import torch
import numpy as np
import sys
import os

# Base Class
from oesd.ours.unified_baseline_utils.adapters.BaseAdapter import BaseAdapter
from oesd.ours.unified_baseline_utils.skill_registry import SkillRegistry

# =========================================================================
# 1. IMPORT YOUR AGENT (Relative Path Fix)
# =========================================================================

# We construct the path relative to the root of the project
DADS_REPO_PATH = os.path.abspath(os.path.join("oesd", "adapters", "baselines", "dads", "scripts"))

# Add to system path if not present
if DADS_REPO_PATH not in sys.path:
    sys.path.append(DADS_REPO_PATH)

# Add the project root to the Python path to resolve imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from oesd.baselines.dads.scripts.dads_trainer import DADSTrainer
from oesd.baselines.dads.scripts.dads import PolicyNet, SkillDynamicsNet, ValueNet

# =========================================================================
# DADS Adapter Class
# =========================================================================
class DADSAdapter(BaseAdapter):
    """
    Adapter for the DADS implementation using DADSTrainer.
    """
    algo_name = "dads"

    def __init__(self, algo_name: str, ckpt_path: str, action_dim: int, save_dir: str, skill_registry: SkillRegistry):
        super().__init__(algo_name, ckpt_path, action_dim, save_dir, skill_registry)

        # Initialize the DADS Architecture
        self.skill_dim = 8  # Hardcoded based on training
        self.cfg = self._create_config(action_dim)

        # Initialize networks
        policy_net = PolicyNet(state_dim=149, skill_dim=self.skill_dim, action_dim=action_dim).to(self.device)
        skill_dynamics_net = SkillDynamicsNet(state_dim=149, skill_dim=self.skill_dim).to(self.device)
        value_net = ValueNet(state_dim=149, skill_dim=self.skill_dim).to(self.device)

        # Initialize trainer
        self.trainer = DADSTrainer(policy_net, skill_dynamics_net, value_net, self.cfg)

    def _create_config(self, action_dim):
        """Create a configuration object for DADSTrainer."""
        class Config:
            state_dim = 149
            num_skills = 8
            action_dim = 6
            policy_lr = 3e-4
            dynamics_lr = 3e-4
            value_lr = 3e-4
            replay_size = 200_000
            batch_size = 128
            device = self.device

        return Config()

    def load_model(self, checkpoint_path: str):
        """Load the model weights from the checkpoint."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'policy_state_dict' in checkpoint:
                self.trainer.policy_net.load_state_dict(checkpoint['policy_state_dict'])
            else:
                self.trainer.policy_net.load_state_dict(checkpoint)
            print(f"DADS Weights loaded successfully from {checkpoint_path}")
        except Exception as e:
            print(f"Error loading DADS weights: {e}")
            raise e

    def preprocess_observation(self, raw_obs):
        """Preprocess the raw observation into the required input format."""
        obs = torch.tensor(raw_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return obs

    def sample_skill(self, rng):
        """Sample a discrete one-hot skill vector."""
        idx = rng.integers(self.skill_dim)
        skill_vec = torch.zeros(self.skill_dim, device=self.device)
        skill_vec[idx] = 1.0
        return skill_vec

    def get_action(self, obs, skill_vec, deterministic=False):
        """Get the action from the policy given the observation and skill vector."""
        with torch.no_grad():
            logits = self.trainer.policy_net(obs, skill_vec)
            action = torch.argmax(logits, dim=-1) if deterministic else torch.multinomial(torch.softmax(logits, dim=-1), 1)
        return action.item()