import torch
from oesd.ours.unified_baseline_utils.skill_registry import SkillRegistry

# ============================================================================
# Base class for all adapters
# ============================================================================

import os
import torch
import numpy as np
from oesd.ours.unified_baseline_utils.skill_registry import SkillRegistry
from BaseAdapter import BaseAdapter   
from oesd.baselines.metra.models import ModelManager


class MetraAdapter(BaseAdapter):
   
    def __init__(self, 
                 algo_name: str, 
                 ckpt_path: str, 
                 action_dim: int,
                 save_dir: str, 
                 skill_registry: SkillRegistry,
                 n_skills: int = 5,
                 latent_dim: int = 2,
                 modelManager=ModelManager(phiNum=3,polNum=3)):
        
        super().__init__(algo_name, ckpt_path, action_dim, save_dir, skill_registry)

        self.n_skills = n_skills
        self.latent_dim = latent_dim

        # ============================
        # 1. CREATE MODELS
        # ============================
        # Expect: modelManager.giveModels() → (PhiNetClass, PolicyClass)

        PhiClass, PolicyClass = modelManager.giveModels()

        self.phi = PhiClass(latent_dim=latent_dim).to(self.device)
        self.policy = PolicyClass(n_skills=n_skills, n_actions=action_dim).to(self.device)
        self.skill_embeddings = torch.nn.Embedding(n_skills, latent_dim).to(self.device)

        # ============================
        # 2. LOAD CHECKPOINTS
        # ============================
        phi_path = os.path.join(ckpt_path, "phi.pth")
        policy_path = os.path.join(ckpt_path, "policy.pth")
        embed_path = os.path.join(ckpt_path, "skillEmbeddings.pth")

        print(f"[METRA] Loading φ from {phi_path}")
        self.phi.load_state_dict(torch.load(phi_path, map_location=self.device))

        print(f"[METRA] Loading policy from {policy_path}")
        self.policy.load_state_dict(torch.load(policy_path, map_location=self.device))

        print(f"[METRA] Loading skill embeddings from {embed_path}")
        self.skill_embeddings.load_state_dict(torch.load(embed_path, map_location=self.device))

        self.phi.eval()
        self.policy.eval()
        self.skill_embeddings.eval()

        # ============================
        # 3. REGISTER SKILLS
        # ============================
        skill_list = []
        for k in range(n_skills):
            emb = self.skill_embeddings(torch.tensor([k], device=self.device)).detach().cpu().numpy()[0]
            skill_list.append(emb)

        self.skill_registry.register_baseline(self.algo_name, skill_list)

        print(f"[METRA] Registered {len(skill_list)} skills in SkillRegistry.")

    # ----------------------------------------------------------------------
    # Unified interface
    # ----------------------------------------------------------------------
    def get_action(self, obs, skill_z, deterministic=False):
        """
        obs: numpy array for one observation
        skill_z: vector from skill_registry (METRA skill embedding)
        deterministic: ignored (METRA uses discrete Q-policy)
        """

        # Ensure skill_z comes from registry
        assert skill_z in self.skill_registry.get_(self.algo_name), \
            f"Skill not recognized or not registered for METRA adapter."

        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        skill_tensor = torch.tensor(skill_z, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Infinite-horizon METRA uses: Q(s, z) → argmax_a
        # The METRA policy takes (state, skill_index); here we use skill embedding directly
        # so we need to adapt: the policy expects skill_idx, but we have embedding
        # → Instead: search for index with matching embedding

        # ----------------------------------
        # Find the nearest skill index
        # ----------------------------------
        with torch.no_grad():
            emb_table = self.skill_embeddings.weight  # [n_skills, latent_dim]
            dist = torch.norm(emb_table - skill_tensor, dim=1)
            skill_idx = torch.argmin(dist)

        with torch.no_grad():
            q_values = self.policy(obs_tensor, skill_idx)
            action = torch.argmax(q_values, dim=1).item()

        return action
