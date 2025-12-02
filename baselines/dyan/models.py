import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HybridEncoder(nn.Module):
    """
    Fuses the Grid Image (Convolution) with the Scalar State (Coordinates + Key).
    This ensures the agent 'sees' the walls BUT also knows exactly where it is.
    """
    def __init__(self, action_dim, feature_dim=256):
        super().__init__()
        
        # 1. Visual Pathway (For seeing walls/doors)
        # Input: (C, 7, 7) -> MiniGrid standard view is 7x7
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=2, stride=1, padding=0), # -> 16x6x6
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=0), # -> 32x5x5
            nn.ReLU(),
            nn.Flatten() # 32*5*5 = 800
        )
        
        # 2. State Pathway (X, Y, HasKey)
        self.state_fc = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU()
        )
        
        # 3. Fusion
        self.fusion = nn.Linear(800 + 32, feature_dim)
        self.ln = nn.LayerNorm(feature_dim)

    def forward(self, img, state_vec):
        x_img = self.conv(img)
        x_state = self.state_fc(state_vec)
        x = torch.cat([x_img, x_state], dim=1)
        x = F.relu(self.fusion(x))
        return self.ln(x)

class Policy(nn.Module):
    def __init__(self, feature_dim, skill_dim, action_dim):
        super().__init__()
        # Input: Features + One-Hot Skill
        self.net = nn.Sequential(
            nn.Linear(feature_dim + skill_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        
    def forward(self, features, skill_vec):
        x = torch.cat([features, skill_vec], dim=1)
        logits = self.net(x)
        return logits

class Discriminator(nn.Module):
    """
    Tries to predict the Skill ID based ONLY on the final state (X, Y, Key).
    We intentionally do NOT give it the image. We want skills to be defined by 
    'Where I am' and 'What I have', not 'What the wall looks like'.
    """
    def __init__(self, skill_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64), # Input is just [x, y, has_key]
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, skill_dim)
        )
        
    def forward(self, state_vec):
        return self.net(state_vec)
