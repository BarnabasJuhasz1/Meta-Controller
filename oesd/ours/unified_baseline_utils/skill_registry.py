import torch
import numpy as np

class SkillRegistry:
    def __init__(self, skill_count_per_algo):
        # list of skill vectors
        self.skills = []
        # number of skills per algorithm
        self.skill_count_per_algo = skill_count_per_algo
        # list of algorithms
        self.algos = []

        # Note: mapping from skill to algo is done by dividing with skill_count

    def register_baseline(self, algo:str, skill_list):
        """
        Registers a model with the given skill vectors
        """
        
        assert len(skill_list) == self.skill_count_per_algo, f"Skill list must have {self.skill_count_per_algo} skills, got {len(skill_list)}"
        
        self.skills.append(skill_list)
        self.algos.append(algo)

        print(f"Registered {algo} with skills: {skill_list}")


    def get_algo_from_skill_idx(self, skill_idx):
        """
        Returns which algo this skill id is corresponding to.
        Intuition: from skill 0 --> skill_count-1 algo A
        from skill k --> 2*skill_count-1 algo B and so on..
        """
        return self.algos[skill_idx//self.kill_count_per_algo]

    def get_skills_belonging_to_algo(self, algo_name):
        """
        Returns the list of skills belonging to the algorithm.
        """
        assert algo_name in self.algos, f"Algorithm {algo_name} not registered"
        start_index = self.algos.index(algo_name)*self.kill_count_per_algo
        end_index = start_index + self.kill_count_per_algo
        return self.skills[start_index:end_index]

    def does_skill_belong_to_algo(self, algo_name, skill_z):
        """
        Returns True if the skill belongs to the algorithm.
        """
        assert algo_name in self.algos, f"Algorithm {algo_name} not registered"
        return skill_z in self.get_skills_belonging_to_algo(algo_name)

    def get_skill_from_skill_idx(self, global_skill_idx):
        """
        Returns the skill vector from the global skill index.
        """
        assert global_skill_idx < len(self.skills), f"Global Skill index {global_skill_idx} out of bounds"
        return self.skills[global_skill_idx]

    def get_algo_skill_from_local_skill_idx(self, algo_name, local_skill_idx):
        """
        Returns the skill vector given the algorithm and the local skill index.
        """
        assert algo_name in self.algos, f"Algorithm {algo_name} not registered"
        assert local_skill_idx < self.kill_count_per_algo, f"Local skill index {local_skill_idx} out of bounds"
        
        return self.skills[self.algos.index(algo_name)+local_skill_idx]


    # def get_action(self, skill_idx, observation):
    #     """
    #     Returns the primitive action from the selected sub-skill.
    #     """
    #     model, skill_list = self.skills[skill_idx]
        
    #     # Convert obs to tensor if necessary
    #     obs_tensor = torch.as_tensor(observation, dtype=torch.float32)
        
    #     # Get primitive action (e.g., MiniGrid move) from the sub-policy
    #     with torch.no_grad():
    #         primitive_action = model.get_action(obs_tensor, z)
            
    #     return primitive_action
    
