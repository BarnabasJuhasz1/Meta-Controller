
import torch
from oesd.ours.unified_baseline_utils.skill_registry import SkillRegistry

# ============================================================================
# Base class for all adapters
# ============================================================================

class BaseAdapter:
    """
    Every adapter must implement:
        - get_action(obs, deterministic)
    """
    def __init__(self, algo_name: str, ckpt_path: str, action_dim: int, save_dir: str, skill_registry: SkillRegistry):
        self.algo_name = algo_name
        self.ckpt_path = ckpt_path
        self.action_dim = action_dim
        self.save_dir = save_dir
        self.skill_registry = skill_registry

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading {self.algo_name} checkpoints from {self.ckpt_path}. Action dim: {self.action_dim}")

    def get_action(self, obs, skill_z, deterministic=False):
        raise NotImplementedError