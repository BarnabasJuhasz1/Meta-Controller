import torch
import torch.nn as nn
import numpy as np
import os
from oesd.ours.unified_baseline_utils.adapters.BaseAdapter import BaseAdapter

# ==============================================================================
# 1. Define the "Legacy" Architecture (Matches checkpoint)
# ==============================================================================
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class LegacyEncoder(nn.Module):
    def __init__(self, feature_dim=256): 
        super().__init__()
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(3, 16, 2)), nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, 2)), nn.ReLU(),
            nn.Flatten()
        )
        self.state_fc = nn.Sequential(layer_init(nn.Linear(3, 32)), nn.ReLU())
        self.fusion = layer_init(nn.Linear(800 + 32, feature_dim))
        self.ln = nn.LayerNorm(feature_dim)
        self.feature_dim = feature_dim

    def forward(self, img, state):
        x = self.conv(img)
        y = self.state_fc(state)
        combined = torch.cat([x, y], dim=1)
        return self.ln(self.fusion(combined))

class LegacyAgent(nn.Module):
    def __init__(self, action_dim, skill_dim):
        super().__init__()
        self.encoder = LegacyEncoder()
        self.actor = nn.Sequential(
            layer_init(nn.Linear(self.encoder.feature_dim + skill_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, action_dim), std=0.01),
        )

    def get_action_and_value(self, img, state, skill):
        features = self.encoder(img, state)
        input_vec = torch.cat([features, skill], dim=1)
        logits = self.actor(input_vec)
        probs = torch.distributions.Categorical(logits=logits)
        return probs.sample(), None, None, None

# ==============================================================================
# 2. The Adapter Class (Robust Version)
# ==============================================================================

class DIAYNAdapter(BaseAdapter):
    def __init__(
        self,
        algo_name: str,
        ckpt_path: str,
        action_dim: int,
        skill_dim: int, 
        save_dir: str = "./",
        skill_registry = None,
        device: str = "cpu"
    ):
        super().__init__(
            algo_name=algo_name, 
            ckpt_path=ckpt_path, 
            action_dim=action_dim, 
            save_dir=save_dir, 
            skill_registry=skill_registry
        )

        self.device = torch.device(device)
        self.skill_dim = skill_dim
        
        # --- INTERNAL SKILL STATE ---
        self.current_skill = torch.zeros(1, skill_dim).to(self.device)
        self.current_skill[0, 0] = 1.0  # Default to Skill 0
        
        print(f"[DIAYNAdapter] Initializing Legacy Agent (Action: {action_dim}, Skill: {skill_dim})")

        self.model = LegacyAgent(action_dim=action_dim, skill_dim=skill_dim)
        self.model.to(self.device)

        if ckpt_path and os.path.exists(ckpt_path):
            print(f"[DIAYNAdapter] Loading checkpoint from {ckpt_path}")
            try:
                checkpoint = torch.load(ckpt_path, map_location=self.device)
            except Exception:
                checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            
            # Check if skill_dim is in checkpoint (from my recent update to agent.py)
            if isinstance(checkpoint, dict) and 'skill_dim' in checkpoint:
                 ckpt_skill_dim = checkpoint['skill_dim']
                 if ckpt_skill_dim != self.skill_dim:
                     print(f"[DIAYNAdapter] WARNING: Checkpoint skill_dim ({ckpt_skill_dim}) != Adapter skill_dim ({self.skill_dim}). Using Checkpoint value.")
                     self.skill_dim = ckpt_skill_dim
                     self.set_skill(0) # Reset current skill vector size
                     # Re-init agent
                     self.model = LegacyAgent(action_dim=action_dim, skill_dim=self.skill_dim)
                     self.model.to(self.device)

            if isinstance(checkpoint, dict):
                if 'encoder' in checkpoint:
                    self.model.encoder.load_state_dict(checkpoint['encoder'])
                    print(" - Encoder loaded.")
                
                if 'policy' in checkpoint:
                    raw_policy = checkpoint['policy']
                    clean_policy = {}
                    for k, v in raw_policy.items():
                        new_key = k.replace("net.", "") 
                        clean_policy[new_key] = v
                    try:
                        self.model.actor.load_state_dict(clean_policy)
                        print(" - Actor (Policy) loaded (fixed 'net.' keys).")
                    except Exception as e:
                        print(f" - Error loading actor: {e}")
                        import traceback
                        traceback.print_exc()
                        # Fallback: try loading directly if clean failed or keys match differently
                        # Fallback: try loading directly if clean failed or keys match differently
                        try:
                             self.model.actor.load_state_dict(raw_policy)
                             print(" - Actor (Policy) loaded directly.")
                        except:
                             pass

                elif 'actor' in checkpoint:
                    self.model.actor.load_state_dict(checkpoint['actor'])
            else:
                self.model.load_state_dict(checkpoint, strict=False)
        else:
            print(f"[DIAYNAdapter] Warning: Checkpoint path '{ckpt_path}' not found!")

        self.model.eval()
        
        # --- REGISTER SKILLS ---
        if self.skill_registry:
            print(f"[DIAYNAdapter] Registering {self.skill_dim} skills for {self.algo_name}...")
            skill_list = []
            for i in range(self.skill_dim):
                z = np.zeros(self.skill_dim, dtype=np.float32)
                z[i] = 1.0
                skill_list.append(z)
            
            # Ensure we match the registry's expectation
            # If registry expects count X and we have Y != X, we might need to pad/truncate or error.
            # But let's assume config aligns them.
            self.skill_registry.register_baseline(self.algo_name, skill_list)

    def set_skill(self, skill_idx):
        """Helper to change the active skill"""
        self.current_skill = torch.zeros(1, self.skill_dim).to(self.device)
        self.current_skill[0, skill_idx % self.skill_dim] = 1.0

    def sample_skill(self, rng: np.random.Generator) -> np.ndarray:
        """Return a skill vector (always dimension = skill_dim)."""
        idx = rng.integers(0, self.skill_dim)
        z = np.zeros(self.skill_dim, dtype=np.float32)
        z[idx] = 1.0
        return z

    def preprocess_observation(self, raw_obs) -> np.ndarray:
        """Convert raw env obs into the model's required input vector."""
        return raw_obs

    def process_obs(self, obs, env=None):
        """
        Process observation to return a flat numpy array (compatible with MetaControllerEnv).
        """
        if isinstance(obs, dict):
            img = obs.get('image', obs.get('visual')) 
        else:
            img = obs
            
        # if img is None:
        #     # Fallback if no image found (e.g. flat obs already?)
        #     return np.zeros(147, dtype=np.float32)

        # returned SHAPE: (147,)
        return img.flatten().astype(np.float32)

    def load_model(self, checkpoint_path: str):
        pass

    # --- ROBUST GET_ACTION ---
    def get_action(
        self,
        obs, 
        skill_vec: np.ndarray = None,
        deterministic: bool = True
    ):
        # CUT THE OBS TO THE SHAPE OF (147,) AS LSD EXPECTS IT
        # obs = obs[:147]

        if skill_vec is not None:
             skill_tensor = torch.tensor(skill_vec, dtype=torch.float32).to(self.device).unsqueeze(0)
        else:
             skill_tensor = self.current_skill

        # 1. Handle Dictionary Observations (Standard Minigrid)
        if isinstance(obs, dict):
            img = obs.get('image')
            if img is None:
                img = obs.get('visual')
            state = obs.get('state', torch.zeros(1, 3).to(self.device))
            
        # 2. Handle Array Observations (Wrapped/Raw Image)
        else:
            img = obs
            state = torch.zeros(1, 3).to(self.device)

        # img = obs.reshape(7, 7, 3)

        # 3. Convert to Tensor
        if not torch.is_tensor(img):
            img = torch.tensor(img, dtype=torch.float32).to(self.device)
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float32).to(self.device)

        # 4. Handle Dimensions
        if img.ndim == 3: 
            img = img.unsqueeze(0)
        
        if state.ndim == 1: 
            state = state.unsqueeze(0)
        
        # Permute (B, H, W, C) -> (B, C, H, W)
        if img.shape[-1] == 3: 
            img = img.permute(0, 3, 1, 2)

        with torch.no_grad():
            action, _, _, _ = self.model.get_action_and_value(img, state, skill_tensor)
            
        return action.item()