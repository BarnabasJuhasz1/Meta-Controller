from __future__ import annotations
import torch
import numpy as np
import sys
import os

# Base Class
from oesd.ours.unified_baseline_utils.adapters.BaseAdapter import BaseAdapter
from oesd.ours.unified_baseline_utils.skill_registry import SkillRegistry

# ============================================================================
# IMPORT PATH FIX
# ============================================================================
# Point to oesd/baselines/dyan
DYAN_REPO_PATH = os.path.abspath(os.path.join("oesd", "baselines", "dyan"))

if DYAN_REPO_PATH not in sys.path:
    sys.path.append(DYAN_REPO_PATH)

try:
    from models import Agent
except ImportError:
    try:
        from oesd.baselines.dyan.models import Agent
    except ImportError:
        print(f"❌ CRITICAL ERROR: Could not import 'models.py' from {DYAN_REPO_PATH}")
        raise

# ============================================================================
# DIAYN Adapter
# ============================================================================

class DIAYNAdapter(BaseAdapter):
    """
    Adapter for PPO-DIAYN.
    """

    def __init__(self, algo_name: str, ckpt_path: str, action_dim: int, save_dir: str, skill_registry: SkillRegistry):
        super().__init__(algo_name, ckpt_path, action_dim, save_dir, skill_registry)

        # 1. Initialize Agent
        self.skill_dim = 8
        self.agent = Agent(action_dim=action_dim, skill_dim=self.skill_dim).to(self.device)

        # 2. Load Weights
        try:
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.agent.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.agent.load_state_dict(checkpoint)
            print(f"✅ DIAYN Weights loaded successfully")
        except Exception as e:
            print(f"❌ Error loading DIAYN weights: {e}")
            raise e

        self.agent.eval()
        
        # 3. REGISTER SKILLS (Critical Step matching RSD)
        # We create 8 one-hot vectors
        skill_list = [self._create_one_hot(i) for i in range(self.skill_dim)]
        
        # Register them so Meta-Controller knows they exist
        self.skill_registry.register_baseline(self.algo_name, skill_list)

    def _create_one_hot(self, idx):
        """Helper to create numpy one-hot vector"""
        vec = np.zeros(self.skill_dim, dtype=np.float32)
        vec[idx] = 1.0
        return vec

    def get_action(self, obs, skill_z, deterministic=False):
        """
        Unified action interface.
        """
        
        # --- 1. PREPARE SKILL ---
        # skill_z comes from registry, so it should be a vector already
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
            # For evaluation, we usually want deterministic actions
            # But earlier you said deterministic=False was better for movement
            # We respect the 'deterministic' flag passed by the visualizer
            
            if deterministic:
                action, _ = self.agent.get_action(img_vec, state_vec, skill_vec, deterministic=True)
            else:
                action, _, _, _ = self.agent.get_action_and_value(img_vec, state_vec, skill_vec)

        return action.item()
