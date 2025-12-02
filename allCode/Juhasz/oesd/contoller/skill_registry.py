import torch
import numpy as np

class SkillRegistry:
    def __init__(self):
        self.skills = [] # List of tuples: (model_func, skill_z)
        
    def register_baseline(self, model_loader, model_path, num_skills, z_dim):
        """
        Loads a baseline model and freezes N discrete skills from it.
        """
        # Assume your loader returns a policy function: action = policy(obs, z)
        policy_net = model_loader(model_path) 
        policy_net.eval() # Freeze weights
        
        # Create a fixed set of latent vectors (Z) to serve as distinct skills
        # For DIAYN/DADS, these are usually one-hot or uniform random samples
        for i in range(num_skills):
            # Example: generating a specific z vector for this skill ID
            z = torch.zeros(z_dim)
            if z_dim > 0:
                # Simple heuristic: spread z values or use one-hot
                z[i % z_dim] = 1.0 
            
            self.skills.append((policy_net, z))
            
    def get_action(self, skill_idx, observation):
        """
        Returns the primitive action from the selected sub-skill.
        """
        model, z = self.skills[skill_idx]
        
        # Convert obs to tensor if necessary
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32)
        
        # Get primitive action (e.g., MiniGrid move) from the sub-policy
        with torch.no_grad():
            primitive_action = model.get_action(obs_tensor, z)
            
        return primitive_action