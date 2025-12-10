import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class HybridEncoder(nn.Module):
    """
    Fuses the Grid Image (Convolution) with the Scalar State (Coordinates + Key).
    """
    def __init__(self):
        super().__init__()
        
        # 1. Visual Pathway (7x7x3 -> 800 features)
        self.cnn = nn.Sequential(
            layer_init(nn.Conv2d(3, 16, 2, stride=1, padding=0)), 
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, 2, stride=1, padding=0)), 
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # 2. State Pathway (X, Y, HasKey -> 64 features)
        self.state_fc = nn.Sequential(
            layer_init(nn.Linear(3, 64)), 
            nn.ReLU()
        )
        
        # Combined Dim: 800 + 64 = 864
        self.feature_dim = 800 + 64

    def forward(self, img, state):
        x_img = self.cnn(img)
        x_state = self.state_fc(state)
        return torch.cat([x_img, x_state], dim=1)

# --- THIS IS THE MISSING CLASS ---
class Agent(nn.Module):
    def __init__(self, action_dim, skill_dim):
        super().__init__()
        self.skill_dim = skill_dim
        self.encoder = HybridEncoder()
        
        # Actor (Policy)
        self.actor = nn.Sequential(
            layer_init(nn.Linear(self.encoder.feature_dim + skill_dim, 256)), 
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)), 
            nn.Tanh(),
            layer_init(nn.Linear(256, action_dim), std=0.01),
        )
        
        # Critic (Value Function)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.encoder.feature_dim + skill_dim, 256)), 
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)), 
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1),
        )

    def get_value(self, img, state, skill):
        features = self.encoder(img, state)
        input_vec = torch.cat([features, skill], dim=1)
        return self.critic(input_vec)

    def get_action_and_value(self, img, state, skill, action=None):
        features = self.encoder(img, state)
        # Concatenate Skill to the features
        input_vec = torch.cat([features, skill], dim=1)
        
        logits = self.actor(input_vec)
        probs = torch.distributions.Categorical(logits=logits)
        
        if action is None:
            action = probs.sample()
            
        return action, probs.log_prob(action), probs.entropy(), self.critic(input_vec)

class Discriminator(nn.Module):
    """
    Predicts Skill ID based ONLY on State (X, Y, Key).
    """
    def __init__(self, skill_dim):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(3, 64)), 
            nn.ReLU(),
            layer_init(nn.Linear(64, 64)), 
            nn.ReLU(),
            layer_init(nn.Linear(64, skill_dim)),
        )
    
    def forward(self, state):
        return self.net(state)
