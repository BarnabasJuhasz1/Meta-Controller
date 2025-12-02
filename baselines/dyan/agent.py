import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models import HybridEncoder, Policy, Discriminator

class DIAYNAgent:
    def __init__(self, env, config):
        # --- FIX START ---
        # Handle "auto" device selection correctly
        device_name = config['training']['device']
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = torch.device(device_name)
        print(f"✅ Agent using device: {self.device}")
        # --- FIX END ---
             
        self.skill_dim = config['agent']['skill_dim']
        self.action_dim = env.action_space.n
        
        # Models
        self.encoder = HybridEncoder(self.action_dim).to(self.device)
        self.policy = Policy(256, self.skill_dim, self.action_dim).to(self.device)
        self.discriminator = Discriminator(self.skill_dim).to(self.device)
        
        # Optimizers
        self.opt_policy = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.policy.parameters()), 
            lr=config['agent']['lr']
        )
        self.opt_disc = torch.optim.Adam(
            self.discriminator.parameters(), 
            lr=config['discriminator']['lr']
        )
        
        self.gamma = config['agent']['gamma']
        self.entropy_coef = config['agent']['entropy_coef']
        self.clip_eps = config['agent']['clip_eps']
        self.update_epochs = config['agent']['update_epochs']

    def get_action(self, obs, skill_vec, deterministic=False):
        img = torch.FloatTensor(obs['image']).unsqueeze(0).to(self.device)
        state = torch.FloatTensor(obs['state']).unsqueeze(0).to(self.device)
        skill = torch.FloatTensor(skill_vec).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.encoder(img, state)
            logits = self.policy(features, skill)
            
            if deterministic:
                return torch.argmax(logits, dim=1).item(), 0.0
            
            probs = F.softmax(logits, dim=1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
        return action.item(), log_prob.item()

    def update(self, memory):
        # Convert memory to tensors
        imgs = torch.FloatTensor(np.array([m['img'] for m in memory])).to(self.device)
        states = torch.FloatTensor(np.array([m['state'] for m in memory])).to(self.device)
        skills = torch.FloatTensor(np.array([m['skill'] for m in memory])).to(self.device)
        actions = torch.LongTensor(np.array([m['action'] for m in memory])).to(self.device)
        old_log_probs = torch.FloatTensor(np.array([m['log_prob'] for m in memory])).to(self.device)
        
        # --- 1. Calculate Rewards (Discriminator Step) ---
        with torch.no_grad():
            # Discriminator predicts skill from STATE only
            disc_logits = self.discriminator(states)
            p_z = F.softmax(disc_logits, dim=1)
            log_p_z = torch.log(p_z + 1e-6)
            
            # DIAYN Reward: log p(z|s) - log(1/k)
            # KEY BONUS: Add reward for holding the key (index 2 is 'carrying')
            # This bridges the gap for DoorKey tasks
            key_bonus = states[:, 2] * 5.0 
            
            prior = np.log(1.0 / self.skill_dim)
            
            intrinsic_rewards = ((log_p_z * skills).sum(dim=1) - prior) + key_bonus
            
        # --- 2. PPO Update (Policy) ---
        returns = []
        R = 0
        for r in reversed(intrinsic_rewards.cpu().numpy()):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.FloatTensor(np.array(returns)).to(self.device)
        
        # Normalize returns for stability
        if returns.std() > 1e-8:
            advantages = (returns - returns.mean()) / (returns.std() + 1e-8)
        else:
            advantages = returns - returns.mean()
        
        for _ in range(self.update_epochs):
            features = self.encoder(imgs, states)
            logits = self.policy(features, skills)
            probs = F.softmax(logits, dim=1)
            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
            
            # Minimize negative loss -> Maximize reward + entropy
            policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
            
            self.opt_policy.zero_grad()
            policy_loss.backward()
            self.opt_policy.step()

        # --- 3. Discriminator Update ---
        disc_pred = self.discriminator(states)
        skill_target = torch.argmax(skills, dim=1)
        disc_loss = F.cross_entropy(disc_pred, skill_target)
        
        self.opt_disc.zero_grad()
        disc_loss.backward()
        self.opt_disc.step()
        
        return intrinsic_rewards.mean().item(), disc_loss.item()

    def save(self, path):
        torch.save({
            'encoder': self.encoder.state_dict(),
            'policy': self.policy.state_dict(),
            'discriminator': self.discriminator.state_dict()
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(ckpt['encoder'])
        self.policy.load_state_dict(ckpt['policy'])
        self.discriminator.load_state_dict(ckpt['discriminator'])