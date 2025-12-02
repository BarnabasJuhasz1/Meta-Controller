
import torch
import torch.nn as nn

import random


from metraTester import DiscreteSkillTester, MetraWrapper


class ModelManager:
    def __init__(self,phiNum=1,polNum=1):
        
        self.phN=phiNum-1
        self.plN = polNum-1

        self.models=[[PhiNet1,PhiNet2,PhiNet3],[MultiSkillPolicyNet1,MultiSkillPolicyNet2,MultiSkillQNetwork]]

    def giveModels(self):

        phi = self.models[0][self.phN]
        pol = self.models[1][self.plN]


        return phi , pol



class PhiNet1(nn.Module):
    def __init__(self, state_dim=2, latent_dim=2, hidden_dim=256, scale=100.0):
        super().__init__()
        self.scale = scale
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
    def forward(self, s):
        return self.scale * self.net(s)


class MultiSkillPolicyNet1(nn.Module):
    def __init__(self, state_dim=2, n_skills=5, n_actions=4, hidden_dim=1024):

        super().__init__()
        self.state_dim = state_dim
        self.n_skills = n_skills
        self.n_actions = n_actions
        
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )     
        
        self.skill_heads = nn.ModuleList([
            nn.Linear(hidden_dim, n_actions) for _ in range(n_skills)
        ])
        
    def forward(self, s, skill_idx):
        
        features = self.feature_net(s)
        
        if skill_idx.dim() == 0:  
            return self.skill_heads[skill_idx](features)
        else: 
            batch_size = s.size(0)
            outputs = []
            for i in range(batch_size):
                skill_i = skill_idx[i].item()
                outputs.append(self.skill_heads[skill_i](features[i:i+1]))
            return torch.cat(outputs, dim=0)
    
    def sample_action(self, s, skill_idx):
        logits = self.forward(s, skill_idx)
        probs = torch.softmax(logits, dim=-1)
        return torch.distributions.Categorical(probs).sample().item()
    

########### 2

class PhiNet2(nn.Module):
    def __init__(self, state_dim=2, latent_dim=2, hidden_dim=256, scale=100.0):
        super().__init__()
        self.scale = scale
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
    def forward(self, s):
        return self.scale * self.net(s)


class MultiSkillPolicyNet2(nn.Module):
    def __init__(self, state_dim=2, n_skills=5, n_actions=4, hidden_dim=1024):
        super().__init__()
        self.state_dim = state_dim
        self.n_skills = n_skills
        self.n_actions = n_actions
        
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )     
        
        self.skill_heads = nn.ModuleList([
            nn.Linear(hidden_dim, n_actions) for _ in range(n_skills)
        ])
        
    def forward(self, s, skill_idx):
        features = self.feature_net(s)
        
        if skill_idx.dim() == 0:  
            return self.skill_heads[skill_idx](features)
        else: 
            batch_size = s.size(0)
            outputs = []
            for i in range(batch_size):
                skill_i = skill_idx[i].item()
                outputs.append(self.skill_heads[skill_i](features[i:i+1]))
            return torch.cat(outputs, dim=0)
    
    def sample_action(self, s, skill_idx):
        logits = self.forward(s, skill_idx)
        probs = torch.softmax(logits, dim=-1)
        return torch.distributions.Categorical(probs).sample().item()
    

### Q network


class PhiNet3(nn.Module):
    def __init__(self, state_dim=2, latent_dim=2, hidden_dim=256, scale=100.0):
        super().__init__()
        self.scale = scale
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
    def forward(self, s):
        return self.scale * self.net(s)


class MultiSkillQNetwork(nn.Module):
    def __init__(self, state_dim=2, n_skills=5, n_actions=4, hidden_dim=1024):
        super().__init__()
        self.state_dim = state_dim
        self.n_skills = n_skills
        self.n_actions = n_actions

        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )     
        
        self.skill_heads = nn.ModuleList([
            nn.Linear(hidden_dim, n_actions) for _ in range(n_skills)
        ])
        
    def forward(self, s, skill_idx):
       
        features = self.feature_net(s)
        
        if skill_idx.dim() == 0:  
            return self.skill_heads[skill_idx](features)
        else: 
            batch_size = s.size(0)
            outputs = []
            for i in range(batch_size):
                skill_i = skill_idx[i].item()
                outputs.append(self.skill_heads[skill_i](features[i:i+1]))
            return torch.cat(outputs, dim=0)
    
    def sample_action(self, s, skill_idx, epsilon=0.1):
        
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            with torch.no_grad():
                q_values = self.forward(s, skill_idx)
                return q_values.argmax(dim=1).item()
