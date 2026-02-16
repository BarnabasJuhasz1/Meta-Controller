# Interface/loader.py

import torch
import numpy as np

# LSD imports
from Algos.LSD.policy import PhiNet as LSDPhiNet
from Algos.LSD.policy import ActorCritic as LSDActorCritic
from Algos.LSD.policy import LSDPolicy
from Algos.LSD.trainer import LSDConfig

# Import DIAYN/RSD policy once teammates implement them
# from Algos.DIAYN.policy import DIAYNPolicy
# from Algos.RSD.policy import RSDPolicy


# =====================================================================
#  Detect algorithm type from checkpoint
# =====================================================================

def _detect_algo(state):
    """
    Infer algorithm type based on checkpoint fields.
    """
    if "algo" in state:
        return state["algo"]

    # Fallback heuristics
    if "phi" in state and "policy" in state:
        return "LSD"

    if "discriminator" in state:
        return "DIAYN"

    if "reconstruction_loss" in state:
        return "RSD"

    raise ValueError("Cannot detect algorithm type from checkpoint.")


# =====================================================================
#  LSD LOADER (fully implemented)
# =====================================================================

def _load_lsd(state, device):
    """
    Rebuild LSD φ-network + actor-critic from checkpoint.
    Returns LSDPolicy.
    """

    # ------------------------------
    # Rebuild config
    # ------------------------------
    cfg_dict = state["cfg"]
    cfg = LSDConfig()
    for k, v in cfg_dict.items():
        setattr(cfg, k, v)

    obs_dim   = cfg.skill_dim  # will be overwritten correctly below
    skill_dim = cfg.skill_dim
    num_skills = cfg.num_skills

    # -------------------------------------------
    # If obs_mean exists, obs_dim = length of mean
    # -------------------------------------------
    if isinstance(state.get("obs_mean"), np.ndarray):
        obs_dim = len(state["obs_mean"])
    else:
        # fallback (should never happen)
        obs_dim = skill_dim

    # ------------------------------
    # Build φ network
    # ------------------------------
    phi = LSDPhiNet(
        obs_dim=obs_dim,
        skill_dim=skill_dim,
        hidden_dim=cfg.phi_hidden_dim
    ).to(device)

    phi.load_state_dict(state["phi"])

    # ------------------------------
    # Build actor-critic network
    # ------------------------------
    # LSD actor input = obs_dim + skill_dim
    # action_dim taken from checkpoint? not stored.
    # BUT we infer it from saved weight shapes.

    policy_state = state["policy"]

    # infer action_dim from the policy head weight shape
    for k, v in policy_state.items():
        if "policy_head.weight" in k:
            action_dim = v.shape[0]
            break
    else:
        raise RuntimeError("Could not infer action_dim for LSD.")

    actor_critic = LSDActorCritic(
        obs_dim=obs_dim,
        skill_dim=skill_dim,
        action_dim=action_dim,
        hidden_dim=cfg.policy_hidden_dim
    ).to(device)

    actor_critic.load_state_dict(policy_state)

    # ------------------------------
    # Build LSDPolicy wrapper
    # ------------------------------
    policy = LSDPolicy(
        actor=actor_critic,
        phi_net=phi,
        num_skills=num_skills,
        device=device
    )

    # Attach normalization stats for external usage
    policy.obs_mean = state.get("obs_mean", None)
    policy.obs_var  = state.get("obs_var", None)
    policy.obs_count = state.get("obs_count", None)

    # This allows visualizer to normalize obs if desired.
    return policy


# =====================================================================
#  DIAYN LOADER (stub for teammates)
# =====================================================================

def _load_diayn(state, device):
    """
    Placeholder until teammates implement DIAYNPolicy / DIAYNTrainer.

    Your teammates MUST:
    - build actor network using cfg
    - load actor weights
    - return DIAYNPolicy(actor, skill_dim, device)
    """

    raise NotImplementedError(
        "DIAYN loader not implemented yet. Teammates must implement "
        "`Algos/DIAYN/policy.py` and extend _load_diayn() accordingly."
    )


# =====================================================================
#  RSD LOADER (stub for teammates)
# =====================================================================

def _load_rsd(state, device):
    raise NotImplementedError(
        "RSD loader not implemented yet. Teammates must implement "
        "`Algos/RSD/policy.py` and extend _load_rsd() accordingly."
    )


# =====================================================================
#  MAIN ENTRY POINT
# =====================================================================

def load_policy(path, device="cpu"):
    """
    Universal policy loader:
    - Detects algorithm
    - Builds correct networks
    - Returns a unified SkillPolicy instance
    """
    state = torch.load(path, map_location=device)

    algo = _detect_algo(state)

    if algo == "LSD":
        return _load_lsd(state, device)

    if algo == "DIAYN":
        return _load_diayn(state, device)

    if algo == "RSD":
        return _load_rsd(state, device)

    raise ValueError(f"Unsupported algorithm type '{algo}' in checkpoint.")
