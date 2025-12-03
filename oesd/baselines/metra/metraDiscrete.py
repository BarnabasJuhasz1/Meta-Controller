import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt
import gymnasium as gym
import torch.nn.functional as F

# Import your existing environment
from example_minigrid import SimpleEnv

from metraTester import DiscreteSkillTester, MetraWrapper


class DiscreteSkillReplayBuffer:
    def __init__(self, capacity=50000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.s_shape = None

    def add(self, s, skill_idx, a, s_next):
        if self.s_shape is None:
            self.s_shape = np.array(s).shape
        self.buffer.append((s, skill_idx, a, s_next))

    def sample(self, batch_size=64):
        batch = random.sample(self.buffer, batch_size)
        
        s_batch = np.zeros((batch_size, *self.s_shape), dtype=np.float32)
        skill_batch = np.zeros(batch_size, dtype=np.int64)
        a_batch = np.zeros(batch_size, dtype=np.int64)
        s_next_batch = np.zeros((batch_size, *self.s_shape), dtype=np.float32)
        
        for i, (s, skill_idx, a, s_next) in enumerate(batch):
            s_batch[i] = s
            skill_batch[i] = skill_idx
            a_batch[i] = a
            s_next_batch[i] = s_next
        
        return (
            torch.from_numpy(s_batch),
            torch.from_numpy(skill_batch),
            torch.from_numpy(a_batch),
            torch.from_numpy(s_next_batch)
        )

    def __len__(self):
        return len(self.buffer)

class DiscreteSkillMetra:
    def __init__(self, env, modelManager, n_skills=5, latent_dim=2,lmbd=30.0, device='cpu'):
        self.n_actions = env.action_space.n
        self.env = MetraWrapper(env)
        self.n_skills = n_skills
        self.latent_dim = latent_dim
        self.device = device


        
        
        self.skill_embeddings = nn.Embedding(n_skills, latent_dim).to(device)

        models = modelManager.giveModels()
        
        self.phi = models[0](latent_dim=latent_dim).to(device)
        self.policy = models[1](
           
            n_skills=n_skills,
            n_actions=self.n_actions
        ).to(device)
        self.targetNetwork = models[1](
            
            n_skills=n_skills, 
            n_actions=self.n_actions
        ).to(device)
        
        #self.skill_embeddings = nn.Embedding(n_skills, latent_dim).to(device)


    
        self.phi_opt = optim.Adam(self.phi.parameters(), lr=1e-4)
        self.q_opt = optim.Adam(
            list(self.policy.parameters()) + list(self.skill_embeddings.parameters()), 
            lr=1e-4
        )
        
        self.targetNetwork.load_state_dict(self.policy.state_dict())
        
        
              
        self.buffer = DiscreteSkillReplayBuffer()
        
        self.lambda_lagrange = torch.tensor(lmbd, requires_grad=False).to(device)

    
        
    def sample_skill(self):
        return random.randint(0, self.n_skills - 1)
    
    def get_skill_embedding(self, skill_idx):

        return self.skill_embeddings(torch.tensor([skill_idx], device=self.device))
    
    def collect_experienceOld(self, steps_per_epoch=50):
        skill_idx = self.sample_skill()
        s = self.env.reset()

        for t in range(steps_per_epoch):
            s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(self.device)
            
           
            skill_tensor = torch.tensor(skill_idx, device=self.device)
            a = self.policy.sample_action(s_tensor, skill_tensor)
            
            s_prev, s_next, done, _ = self.env.step(a)
            
           
            self.buffer.add(s_prev, skill_idx, a, s_next)
            
            s = s_next
            
            if done:
                s = self.env.reset()

                skill_idx = self.sample_skill()
    
    def collect_experience(self, steps_per_epoch=50, epsilon=0.1):
        
        total_steps = 0
        while total_steps < steps_per_epoch:
            
            skill_idx = self.sample_skill()
            s = self.env.reset()
            episode_steps = 0
            done = False  
            
            while not done and (total_steps + episode_steps) < steps_per_epoch:
                s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(self.device)
                skill_tensor = torch.tensor(skill_idx, device=self.device)
                
                
                a = self.policy.sample_action(s_tensor, skill_tensor, epsilon)
                
               
                s_next, reward_env, done, _ = self.env.step(a)  
                
                
                self.buffer.add(s, skill_idx, a, s_next)
                
                # Update state
                s = s_next
                episode_steps += 1
            
            total_steps += episode_steps
    
    def update(self, batch_size=64, eps=1e-3):
        if len(self.buffer) < batch_size:
            return self.lambda_lagrange
        
        s, skill_idx, a, s_next = self.buffer.sample(batch_size)
        
        s = s.to(self.device)
        s_next = s_next.to(self.device)
        a = a.to(self.device)
        skill_idx = skill_idx.to(self.device)
        
       
        z = self.skill_embeddings(skill_idx)
        
        
        phi_s = self.phi(s)
        phi_s_next = self.phi(s_next)


        all_skill_indices = torch.arange(self.n_skills, device=self.device)
        all_z = self.skill_embeddings(all_skill_indices)
        z_mean = all_z.mean(dim=0, keepdim=True)  
        z_centered = z - z_mean  
        reward = torch.sum((phi_s_next - phi_s) * z_centered, dim=1)

        


        pos_current = s[:, :2]  
        pos_next = s_next[:, :2]
        temp_dist = torch.sum(torch.abs(pos_next - pos_current), dim=1)

        phi_diff_norm = torch.norm(phi_s_next - phi_s, dim=1, p=2)
        constraint_violation = torch.clamp(phi_diff_norm - temp_dist, min=0)

    

        q_values = self.policy(s, skill_idx)  
        q_value = q_values[range(batch_size), a]  
        
        with torch.no_grad():
            
            next_q_values = self.targetNetwork(s_next, skill_idx)  # [batch_size, n_actions]
            next_q_value = next_q_values.max(dim=1)[0]  # max_a Q(s', z, a)
            target_q = reward + 0.99 * next_q_value  # with discount factor
        
        q_loss = F.mse_loss(q_value, target_q)

        
        constraint_loss = torch.mean(
            self.lambda_lagrange * constraint_violation + 
            0.5 * eps * constraint_violation**2
        )
        
        # 3. Total loss and optimization
        total_loss = q_loss + constraint_loss
        
        self.phi_opt.zero_grad()
        self.q_opt.zero_grad()
        total_loss.backward()
        self.phi_opt.step()
        self.q_opt.step()
        
        
        with torch.no_grad():
            for target_param, param in zip(self.targetNetwork.parameters(), 
                                         self.policy.parameters()):
                target_param.data.copy_(0.995 * target_param.data + 0.005 * param.data)
        
   
        with torch.no_grad():
            lambda_update = constraint_violation.mean()
            self.lambda_lagrange += 0.01 * lambda_update
            self.lambda_lagrange = torch.clamp(self.lambda_lagrange, min=0.0)


        return self.lambda_lagrange





        
      
    
    def train(self, num_epochs=5000, steps_per_epoch=15, log_interval=100):
        
        for epoch in range(num_epochs):
            self.collect_experience(steps_per_epoch)
            self.lambda_lagrange = self.update()
            
            if epoch % log_interval == 0:
                print(f"[Epoch {epoch}] lambda={self.lambda_lagrange.item():.3f}, "
                      f"buffer={len(self.buffer)}")
    

    def save_models(self, path_prefix="D:\\dionigi\\Documents\\Python scripts\\Open-Ended-Skill-Discovery\\scripts\\metra\\models"):
        torch.save(self.phi.state_dict(), f"{path_prefix}phi.pth")
        torch.save(self.policy.state_dict(), f"{path_prefix}policy.pth")
        torch.save(self.skill_embeddings.state_dict(), f"{path_prefix}skillEmbeddings.pth")
    
    def load_models(self, path_prefix="D:\\dionigi\\Documents\\Python scripts\\Open-Ended-Skill-Discovery\\scripts\\metra\\models"):
        self.phi.load_state_dict(torch.load(f"{path_prefix}phi.pth", map_location=self.device))
        self.policy.load_state_dict(torch.load(f"{path_prefix}policy.pth", map_location=self.device))
        self.skill_embeddings.load_state_dict(torch.load(f"{path_prefix}skillEmbeddings.pth", map_location=self.device))

