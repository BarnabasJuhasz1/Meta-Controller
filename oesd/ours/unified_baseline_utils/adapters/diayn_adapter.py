from __future__ import annotations
import torch
import numpy as np
import sys
import os

# Import the Base Adapter
from oesd.ours.unified_baseline_utils.adapters.BaseAdapter import BaseAdapter
from oesd.ours.unified_baseline_utils.skill_registry import SkillRegistry

# --- IMPORT YOUR AGENT ---
#sys.path.append(os.path.abspath("continual_diayn"))

try:
    from continual_diayn.models import Agent
except ImportError:
    try:
        from models import Agent
    except:
        print("Could not import 'models.py' from continual_diayn. Please check folder location.")

# ============================================================================
# DIAYN Adapter
# ============================================================================

class DIAYNAdapter(BaseAdapter):
    """
    Adapter for the PPO-DIAYN implementation.
    Bridging the gap between Meta-Controller (Flat Obs) and PPO Agent (Hybrid Obs).
    """

    def __init__(self, algo_name: str, ckpt_path: str, action_dim: int, save_dir: str, skill_registry: SkillRegistry):
        super().__init__(algo_name, ckpt_path, action_dim, save_dir, skill_registry)

        # 1. Initialize the PPO Architecture
        # Hardcoded 8 skills based on your training config
        self.skill_dim = 8
        self.agent = Agent(action_dim=action_dim, skill_dim=self.skill_dim).to(self.device)

        # 2. Load Weights
        try:
            # Handle different saving conventions
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.agent.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.agent.load_state_dict(checkpoint)
            print(f"DIAYN Weights loaded successfully from {ckpt_path}")
        except Exception as e:
            print(f"Error loading DIAYN weights: {e}")
            raise e

        self.agent.eval()

    def get_action(self, obs, skill_z, deterministic=False):
        """
        Args:
            obs: Flat numpy array from MetaEnv [Image(147), Direction(1), Carrying(1), X(1), Y(1)]
            skill_z: The skill vector (or index) from the registry
            deterministic: Boolean
        """
        
        # --- 1. PREPARE SKILL ---
        # Convert skill_z to One-Hot Tensor
        if isinstance(skill_z, int) or (isinstance(skill_z, torch.Tensor) and skill_z.numel() == 1):
            # If we get an index, convert to one-hot
            idx = int(skill_z)
            skill_vec = torch.zeros(1, self.skill_dim).to(self.device)
            skill_vec[0][idx] = 1.0
        else:
            # If we get a vector, ensure shape
            skill_vec = torch.as_tensor(skill_z, device=self.device, dtype=torch.float32)
            if skill_vec.ndim == 1:
                skill_vec = skill_vec.unsqueeze(0)

        # --- 2. PREPARE OBSERVATION (The "Un-Flattening") ---
        # Convert to tensor
        if not torch.is_tensor(obs):
            obs_t = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        else:
            obs_t = obs.to(self.device)
            
        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0)

        # RECONSTRUCT STATE: [X, Y, Key]
        # Based on MetaEnv wrapper: X is -2, Y is -1, Carrying is -3
        x = obs_t[:, -2]
        y = obs_t[:, -1]
        key = obs_t[:, -3]
        state_vec = torch.stack([x, y, key], dim=1)

        # RECONSTRUCT IMAGE: First 147 elements (7*7*3)
        # Minigrid provides (7, 7, 3). Flattened = 147.
        # Your PPO Agent expects (3, 7, 7) -> Channels First.
        img_flat = obs_t[:, :147]
        img_reshaped = img_flat.view(-1, 7, 7, 3) # Restore HWC
        img_vec = img_reshaped.permute(0, 3, 1, 2) # Convert to CHW

        # --- 3. FORWARD PASS ---
        with torch.no_grad():
            # Your agent.get_action_and_value expects (img, state, skill)
            action, _, _, _ = self.agent.get_action_and_value(img_vec, state_vec, skill_vec)

        # Return standard integer
        return action.item()
