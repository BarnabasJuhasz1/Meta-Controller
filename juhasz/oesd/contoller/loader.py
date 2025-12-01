import numpy as np
import torch
import sys
import os

def load_RSD_checkpoint(checkpoint_dir, epoch):
    option_policy_path = os.path.join(checkpoint_dir, f'option_policy{epoch}.pt')
    traj_encoder_path = os.path.join(checkpoint_dir, f'traj_encoder{epoch}.pt')
    sample_z_policy_path = os.path.join(checkpoint_dir, f'SampleZPolicy{epoch}.pt')

    print(f"Loading checkpoints from {checkpoint_dir} at epoch {epoch}...")
    option_policy_ckpt = torch.load(option_policy_path, map_location='cpu', weights_only=False)
    traj_encoder_ckpt = None
    sample_z_ckpt = None

    if os.path.exists(traj_encoder_path):
        traj_encoder_ckpt = torch.load(traj_encoder_path, map_location='cpu', weights_only=False)
    else:
        print(f"Warning: Missing {traj_encoder_path}")

    if os.path.exists(sample_z_policy_path):
        sample_z_ckpt = torch.load(sample_z_policy_path, map_location='cpu', weights_only=False)
    else:
        print(f"Warning: Missing {sample_z_policy_path}")

    return option_policy_ckpt, traj_encoder_ckpt, sample_z_ckpt

def load_DADS_checkpoint(checkpoint_dir, epoch):
    pass