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

    def __init__(self, algo_name: str, algo_color: str, ckpt_path: str, action_dim: int, skill_dim: int, save_dir: str, skill_registry: SkillRegistry):
        super().__init__(algo_name, algo_color, ckpt_path, action_dim, save_dir, skill_registry)

        self.skill_dim = skill_dim

        # Initialize the DADS Architecture
        self.cfg = self._create_config(action_dim)

        # Initialize networks
        policy_net = PolicyNet(state_dim=149, skill_dim=self.skill_dim, action_dim=action_dim).to(self.device)
        skill_dynamics_net = SkillDynamicsNet(state_dim=149, skill_dim=self.skill_dim).to(self.device)
        value_net = ValueNet(state_dim=149, skill_dim=self.skill_dim).to(self.device)

        # Initialize trainer
        self.trainer = DADSTrainer(policy_net, skill_dynamics_net, value_net, self.cfg)

        # --- REGISTER SKILLS ---
        if self.skill_registry:
            print(f"[DADSAdapter] Registering {self.skill_dim} skills for {self.algo_name}...")
            skill_list = []
            for i in range(self.skill_dim):
                z = np.zeros(self.skill_dim, dtype=np.float32)
                z[i] = 1.0
                skill_list.append(z)
            
            # Ensure we match the registry's expectation
            # If registry expects count X and we have Y != X, we might need to pad/truncate or error.
            # But let's assume config aligns them.
            self.skill_registry.register_baseline(self.algo_name, skill_list)


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

    # def process_obs(self, raw_obs, env=None):
    #     """Preprocess the raw observation into the required input format."""
    #     obs = torch.tensor(raw_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
    #     return obs
    def process_obs(self, obs, env):
        if isinstance(obs, tuple):
            obs = obs[0]
        
        # 1. Flatten and normalize image
        image = obs["image"].astype(np.float32) / 255.0
        
        # 2. Normalize direction (0-3 becomes 0.0-1.0)
        direction = np.array([obs["direction"] / 3.0], dtype=np.float32)

        # 3. Add Carrying (Binary)
        # Accessing .unwrapped is safer in case of other wrappers
        # env_base = self.env.unwrapped 
        carrying_val = 1.0 if env.carrying is not None else 0.0
        carrying = np.array([carrying_val], dtype=np.float32)

        # Concatenate everything: [Image flat, Direction, Carrying, PosX, PosY]
        obs = np.concatenate([
            image.flatten(), 
            direction, 
            carrying, 
        ], axis=0)
        return obs
        # return torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

    def sample_skill(self, rng):
        """Sample a discrete one-hot skill vector."""
        idx = rng.integers(self.skill_dim)
        skill_vec = torch.zeros(self.skill_dim, device=self.device)
        skill_vec[idx] = 1.0
        return skill_vec

    def get_action(self, obs, skill_z, deterministic=False):
        # (149,) (8,)
        #print(obs.shape, skill_z.shape)

        """Get the action from the policy given the observation and skill vector."""
        with torch.no_grad():
            # obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            obs = torch.from_numpy(obs).to(self.device).unsqueeze(0)
            skill_z = torch.tensor(skill_z, dtype=torch.float32, device=self.device).unsqueeze(0)

            logits = self.trainer.policy_net(obs, skill_z)
            action = torch.argmax(logits, dim=-1) if deterministic else torch.multinomial(torch.softmax(logits, dim=-1), 1)
        return action.item()

    