from __future__ import annotations
import torch
import numpy as np
import sys
import os

# Base Class
from oesd.ours.unified_baseline_utils.adapters.BaseAdapter import BaseAdapter
from oesd.ours.unified_baseline_utils.skill_registry import SkillRegistry

# --- 1. IMPORT YOUR AGENT ---
# We point to your Documents folder to find 'models.py'
YOUR_CODE_DIR = r"C:\Users\sajay\Documents\continual_diayn"
if YOUR_CODE_DIR not in sys.path:
    sys.path.append(YOUR_CODE_DIR)

try:
    from models import Agent
except ImportError:
    print(f"❌ Error: Could not import 'models' from {YOUR_CODE_DIR}")

# ============================================================================
# DIAYN Adapter
# ============================================================================

class DIAYNAdapter(BaseAdapter):
    """
    Adapter for PPO-DIAYN.
    Matches the style of LSDAdapter.
    """

    def __init__(self, algo_name: str, ckpt_path: str, action_dim: int, save_dir: str, skill_registry: SkillRegistry):
        super().__init__(algo_name, ckpt_path, action_dim, save_dir, skill_registry)

        # 1. Initialize Agent Architecture
        self.skill_dim = 8  # Hardcoded based on your training
        self.agent = Agent(action_dim=action_dim, skill_dim=self.skill_dim).to(self.device)

        # 2. Load Weights
        try:
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.agent.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.agent.load_state_dict(checkpoint)
            print(f"✅ DIAYN Weights loaded from {ckpt_path}")
        except Exception as e:
            print(f"❌ Failed to load DIAYN: {e}")
            raise e

        self.agent.eval()

    def get_action(self, obs, skill_z, deterministic=False):
        """
        Unified action interface.
        """
        # --- Prepare Skill ---
        # Handle int index (0-7) or tensor input
        if isinstance(skill_z, (int, np.integer)):
            idx = int(skill_z)
            skill_vec = torch.zeros(1, self.skill_dim).to(self.device)
            skill_vec[0][idx] = 1.0
        elif isinstance(skill_z, torch.Tensor):
            if skill_z.numel() == 1: # It's an index inside a tensor
                idx = int(skill_z.item())
                skill_vec = torch.zeros(1, self.skill_dim).to(self.device)
                skill_vec[0][idx] = 1.0
            else: # It's already a vector
                skill_vec = skill_z.to(self.device)
                if skill_vec.ndim == 1: skill_vec = skill_vec.unsqueeze(0)
        else:
            # Fallback for numpy arrays
            skill_vec = torch.tensor(skill_z).float().to(self.device)
            if skill_vec.ndim == 1: skill_vec = skill_vec.unsqueeze(0)

        # --- Prepare Observation ---
        # Un-flatten the vector back into Image + State
        if not torch.is_tensor(obs):
            obs_t = torch.tensor(obs, dtype=torch.float32).to(self.device)
        else:
            obs_t = obs.to(self.device)
        
        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0)

        # Logic based on MetaEnv wrapper structure:
        # [Image(147), Direction(1), Carrying(1), X(1), Y(1)]
        x = obs_t[:, -2]
        y = obs_t[:, -1]
        key = obs_t[:, -3]
        state_vec = torch.stack([x, y, key], dim=1)

        img_flat = obs_t[:, :147]
        img_vec = img_flat.view(-1, 3, 7, 7) # Reshape 147 -> 3x7x7

        # --- Forward Pass ---
        with torch.no_grad():
            action, _, _, _ = self.agent.get_action_and_value(img_vec, state_vec, skill_vec)

        return action.item()
