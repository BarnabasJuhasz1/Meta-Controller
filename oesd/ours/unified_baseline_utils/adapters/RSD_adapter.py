from __future__ import annotations
from dataclasses import dataclass
from typing import Any

import torch
import numpy as np
from oesd.ours.unified_baseline_utils.skill_registry import SkillRegistry
from oesd.ours.unified_baseline_utils.adapters.BaseAdapter import BaseAdapter

import sys
import os
import oesd

# --- STEP 1: Define Paths ---
oesd_root = os.path.dirname(oesd.__file__)
garage_src_path = os.path.join(oesd_root, "baselines", "RSD", "garaged", "src")
rsd_path = os.path.join(oesd_root, "baselines", "RSD")

# --- STEP 2: Update sys.path ---
# Allow Python to find 'garage'
if garage_src_path not in sys.path:
    sys.path.append(garage_src_path)

# Allow Python to find 'global_context' AND 'dowel_wrapper'
if rsd_path not in sys.path:
    sys.path.append(rsd_path)

# --- STEP 3: Handle dowel_wrapper assertion ---
# 1. Unload dowel if it exists (to satisfy the wrapper's assertion)
if 'dowel' in sys.modules:
    del sys.modules['dowel']

# 2. CHANGE IS HERE: Import it using the SHORT name.
#    This works because 'rsd_path' is in sys.path now.
#    This registers 'dowel_wrapper' in sys.modules so the checkpoint finds it.
import dowel_wrapper


# FIXING GARAGE IMPORT PROBLEM
# import sys
# import os
# import importlib

# try:
#     # Now this simple import must work natively
#     import oesd.baselines.RSD.garaged.src.garage
#     print(f"✅ 'garage' is now importable from: {garage.__file__}")
# except ImportError:
#     print("❌ Still cannot import garage directly.")


# # 1. Import the nested garage module using the full path
# import oesd.baselines.RSD.garaged.src.garage as nested_garage

# # 2. Get the physical path to the 'src' directory
# # nested_garage.__file__ gives .../src/garage/__init__.py
# # dirname gives .../src/garage
# # dirname again gives .../src
# garage_package_dir = os.path.dirname(os.path.dirname(nested_garage.__file__))

# # 3. Add that 'src' directory to sys.path
# if garage_package_dir not in sys.path:
#     sys.path.append(garage_package_dir)

# # 4. VERIFICATION (Optional)
# try:
#     # Now this simple import must work natively
#     import garage
#     print(f"✅ 'garage' is now importable from: {garage.__file__}")
# except ImportError:
#     print("❌ Still cannot import garage directly.")

# ============================================================================
# RSD Adapter
# ============================================================================

class RSDAdapter(BaseAdapter):
    """
    Adapter for your RSD implementation.
    Provides a uniform get_action() interface.
    """
    def __init__(self, algo_name: str, algo_color: str, ckpt_path: str, action_dim: int, save_dir: str, skill_registry: SkillRegistry):
        # also makes sure self.device and self.skill_registry are set
        super().__init__(algo_name, algo_color, ckpt_path, action_dim, save_dir, skill_registry)
    
        self.option_policy_ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

        self.discrete = self.option_policy_ckpt['discrete']
        self.dim_option = self.option_policy_ckpt['dim_option']
        # self.dim_option = 8
        
        skill_list = [self.init_skill_vector(i, unit_length=True) for i in range(skill_registry.skill_count_per_algo)]
        # register the skills with the skill registry
        self.skill_registry.register_baseline(self.algo_name, skill_list)
        # from now on, only skill_registry exposes the skills of this adapter
        # and only skill_registry is used to get the skills to ensure consistency

        self.option_policy = self.option_policy_ckpt['policy']
        self.option_policy.eval()
        self.option_policy = self.option_policy.to(self.device)

    def init_skill_vector(self, k, unit_length=True):
        """
        Helper function to initialize skill vectors
        """
        if self.discrete:
            return np.eye(self.dim_option)[k]

        if self.dim_option == 2:
            num_skills = 8
            angle = (k % num_skills) * 2 * np.pi / num_skills
            r = 1.0 if unit_length else 1.5
            return np.array([r * np.cos(angle), r * np.sin(angle)])

        # Use basis vectors for diversity in higher dimensions
        if k < self.dim_option:
            v = np.zeros(self.dim_option)
            v[k] = 1.0
        elif k < 2 * self.dim_option:
            v = np.zeros(self.dim_option)
            v[k - self.dim_option] = 1.0
        else:
            np.random.seed(k)
            v = np.random.randn(self.dim_option)

        if unit_length:
            v = v / (np.linalg.norm(v) + 1e-8)
        return v

    def get_action(self, obs, skill_z, deterministic=False):
        """
        Unified action interface:
        """
        assert self.skill_registry.does_skill_belong_to_algo(self.algo_name, skill_z), f"skill_z must be in the list of skills for this algo ({self.algo_name})!"

        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        skill_tensor = torch.from_numpy(skill_z).float().unsqueeze(0).to(self.device)


        with torch.no_grad():

            obs_tensor = self.option_policy.process_observations(obs_tensor)
            concat_obs = torch.cat([obs_tensor, skill_tensor], dim=1)
            # I disabled processing inside so we process unified in the env wrapper
            # if hasattr(self.option_policy, 'process_observations'):
            #     # print("observations ARE processed inside get_action")
            #     processed_obs = self.option_policy.process_observations(obs_tensor)
            # else:
            #     # print("observations are NOT processed inside get_action")
            #     processed_obs = obs_tensor
            # concat_obs = torch.cat([processed_obs, skill_tensor], dim=1)

            with torch.no_grad():
                dist, _ = self.option_policy(concat_obs)
                action = dist.mean.cpu().numpy()[0]

        return action

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

        # 4. Add Agent Position (Normalized)
        # We divide by width/height to keep inputs within [0, 1] range
        agent_x = env.agent_pos[0] / env.width
        agent_y = env.agent_pos[1] / env.height
        position = np.array([agent_x, agent_y], dtype=np.float32)

        # debug print
        # if carrying_val > 0:
        #     print(f"AGENT CARRYING at {self._env.agent_pos}")
        
        # Concatenate everything: [Image flat, Direction, Carrying, PosX, PosY]
        return np.concatenate([
            image.flatten(), 
            direction, 
            carrying, 
            # position
        ], axis=0)

        # returned SHAPE: (149,)