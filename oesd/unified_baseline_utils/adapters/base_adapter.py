from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import torch


class BaseAdapter(ABC):
    """
    The minimal interface that all algorithms must expose so the
    visualizer and env wrapper never depend on algorithm-specific logic.
    """

    def __init__(self, action_dim: int, skill_dim: int):
        self.action_dim = action_dim
        self.skill_dim = skill_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------------------------------------------------
    # 1) LOAD MODEL
    # ---------------------------------------------------------------------
    @abstractmethod
    def load_model(self, checkpoint_path: str):
        """Load a model from disk and return the algorithm-specific model."""
        pass

    # ---------------------------------------------------------------------
    # 2) OBSERVATION PREPROCESSING
    # ---------------------------------------------------------------------
    @abstractmethod
    def preprocess_observation(self, raw_obs) -> np.ndarray:
        """Convert raw env obs into the model's required input vector."""
        pass

    # ---------------------------------------------------------------------
    # 3) SKILL SAMPLING
    # ---------------------------------------------------------------------
    @abstractmethod
    def sample_skill(self, rng: np.random.Generator) -> np.ndarray:
        """Return a skill vector (always dimension = skill_dim)."""
        pass

    # ---------------------------------------------------------------------
    # 4) ACTION SELECTION
    # ---------------------------------------------------------------------
    @abstractmethod
    def get_action(
        self,
        model,
        obs_vec: np.ndarray,
        skill_vec: np.ndarray,
        deterministic: bool = False,
    ):
        """Return an env action from model(obs, skill)."""
        pass
