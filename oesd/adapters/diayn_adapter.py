from __future__ import annotations
import torch
import numpy as np
import sys
import os

# Import the Base Adapter
from oesd.ours.unified_baseline_utils.adapters.BaseAdapter import BaseAdapter
from oesd.ours.unified_baseline_utils.skill_registry import SkillRegistry

# ============================================================================
# IMPORT PATH FIX
# ============================================================================
# We add the new location to the system path so Python can find 'models.py'
# Current Path: oesd/baselines/dyan

# 1. Define the new path
# (Using os.path.join ensures it works on Windows and Linux)
DIAN_PATH = os.path.abspath(os.path.join("oesd", "baselines", "dyan"))

# 2. Add to Python Path
if DIAN_PATH not in sys.path:
    sys.path.append(DIAN_PATH)

# 3. Import the Agent
try:
    # Try importing as a package first (Best Practice)
    from oesd.baselines.dyan.models import Agent
except ImportError:
    try:
        # Fallback: Try importing directly since we added it to sys.path
        from models import Agent
    except ImportError:
        print(f"CRITICAL ERROR: Could not import 'Agent' from {DIAN_PATH}")
        print(f"Please ensure 'models.py' exists inside {DIAN_PATH}")
        print(f"And ensure the folder has an empty '__init__.py' file.")

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
        self.skill_dim = 8 # Hardcoded based on your training
        
        # Initialize Agent using the imported class
        self.agent = Agent(action_dim=action_dim, skill_dim=self.skill_dim).to(self.device)

        # 2. Load Weights
        try:
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            
            # Handle different saving structures
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
            obs: Flat numpy array from MetaEnv
            skill_z: The skill vector (or index)
        """
        
        # --- 1. PREPARE SKILL ---
        if isinstance(skill_z, int) or (isinstance(skill_z, torch.Tensor) and skill_z.numel() == 1):
            idx = int(skill_z)
            skill_vec = torch.zeros(1, self.skill_dim).to(self.device)
            skill_vec[0][idx] = 1.0
        else:
            skill_vec = torch.as_tensor(skill_z, device=self.device, dtype=torch.float32)
            if skill_vec.ndim == 1:
                skill_vec = skill_vec.unsqueeze(0)

        # --- 2. PREPARE OBSERVATION ---
        if not torch.is_tensor(obs):
            obs_t = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        else:
            obs_t = obs.to(self.device)
            
        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0)

        # Reconstruct State & Image from Flat Observation
        # [Image(147), Direction(1), Carrying(1), X(1), Y(1)]
        x = obs_t[:, -2]
        y = obs_t[:, -1]
        key = obs_t[:, -3]
        state_vec = torch.stack([x, y, key], dim=1)

        img_flat = obs_t[:, :147]
        img_reshaped = img_flat.view(-1, 7, 7, 3) 
        img_vec = img_reshaped.permute(0, 3, 1, 2) # Channel First

        # --- 3. FORWARD PASS ---
        with torch.no_grad():
            action, _, _, _ = self.agent.get_action_and_value(img_vec, state_vec, skill_vec)

        return action.item()
