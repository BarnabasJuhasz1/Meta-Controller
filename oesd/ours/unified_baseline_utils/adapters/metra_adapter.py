import torch
from oesd.ours.unified_baseline_utils.skill_registry import SkillRegistry

# ============================================================================
# Base class for all adapters
# ============================================================================

import os
import torch
import numpy as np
from oesd.ours.unified_baseline_utils.skill_registry import SkillRegistry
from oesd.ours.unified_baseline_utils.adapters.BaseAdapter import BaseAdapter   
from oesd.baselines.metra.models import ModelManager


class MetraAdapter(BaseAdapter):
   
    def __init__(self, 
                 algo_name: str,
                 algo_color: str,
                 ckpt_path: str, 
                 action_dim: int,
                 save_dir: str, 
                 skill_registry: SkillRegistry,
                 skill_dim: int = 5,
                 latent_dim: int = 2,
                 modelManager=ModelManager(phiNum=3,polNum=3)):
        
        super().__init__(algo_name, algo_color, ckpt_path, action_dim, save_dir, skill_registry)

        self.skill_dim = skill_dim
        self.latent_dim = latent_dim

        # ============================
        # 1. CREATE MODELS
        # ============================
        # Expect: modelManager.giveModels() → (PhiNetClass, PolicyClass)

        

        PhiClass, PolicyClass = modelManager.giveModels()

        self.phi = PhiClass(latent_dim=latent_dim).to(self.device)
        self.policy = PolicyClass(n_skills=skill_dim, n_actions=action_dim).to(self.device)
        self.skill_embeddings = torch.nn.Embedding(skill_dim, latent_dim).to(self.device)

        # ============================
        # 2. LOAD CHECKPOINTS
        # ============================
        phi_path = os.path.join(ckpt_path, "phiDiscreteSkill.pth")
        policy_path = os.path.join(ckpt_path, "policyDiscreteSkill.pth")
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
        for k in range(skill_dim):
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
        assert self.skill_registry.does_skill_belong_to_algo(self.algo_name, skill_z), f"skill_z must be in the list of skills for this algo ({self.algo_name})!"

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

    def process_obs(self, obs, env):

        image = obs["image"].astype(np.float32)

        if isinstance(obs, dict):
            if 'agent_pos' in obs:
                agent_position = np.array(obs['agent_pos'], dtype=np.float32)
            elif 'observation' in obs and hasattr(obs['observation'], 'agent_pos'):
                agent_position = np.array(obs['observation'].agent_pos, dtype=np.float32)
            else:
                agent_position = np.array([0, 0], dtype=np.float32)
        
        return np.concatenate([
            image.flatten(),  
            agent_position
        ], axis=0)

        # returned SHAPE: (147,)
