import torch
import numpy as np

class SkillRegistry:
    def __init__(self, skill_count_per_algo):
        # list of skill vectors
        self.bag_of_skills = []
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
        
        self.bag_of_skills.extend(skill_list)
        self.algos.append(algo.upper())
        print(f"Registered {algo.upper()} with skills: {skill_list}")

    def get_algo_color(self, skill_idx):
        """
        Returns the color of the algorithm corresponding to the skill.
        """
        return self.algo_colors[skill_idx//self.skill_count_per_algo]

    def get_algo_from_skill_idx(self, skill_idx):
        """
        Returns which algo this skill id is corresponding to.
        Intuition: from skill 0 --> skill_count-1 algo A
        from skill k --> 2*skill_count-1 algo B and so on..
        """
        return self.algos[skill_idx//self.skill_count_per_algo]

    def get_skills_belonging_to_algo(self, algo_name):
        """
        Returns the list of skills belonging to the algorithm.
        """
        assert algo_name in self.algos, f"Algorithm {algo_name} not registered"
        start_index = self.algos.index(algo_name)*self.skill_count_per_algo
        end_index = start_index + self.skill_count_per_algo
        return self.bag_of_skills[start_index:end_index]

    def does_skill_belong_to_algo(self, algo_name, skill_z):
        """
        Returns True if the skill belongs to the algorithm.
        """
        assert algo_name in self.algos, f"Algorithm {algo_name} not registered"
        skills = self.get_skills_belonging_to_algo(algo_name)
        for s in skills:
            if np.array_equal(s, skill_z):
                return True
        return False

    def get_skill_from_skill_idx(self, global_skill_idx):
        """
        Returns the skill vector from the global skill index.
        """
        assert global_skill_idx < len(self.bag_of_skills), f"Global Skill index {global_skill_idx} out of bounds"
        return self.bag_of_skills[global_skill_idx]

    def get_algo_skill_from_local_skill_idx(self, algo_name, local_skill_idx):
        """
        Returns the skill vector given the algorithm and the local skill index.
        """
        assert algo_name in self.algos, f"Algorithm {algo_name} not registered"
        assert local_skill_idx < self.skill_count_per_algo, f"Local skill index {local_skill_idx} out of bounds"
        
        return self.bag_of_skills[self.algos.index(algo_name)+local_skill_idx]

    def get_algo_and_skill_from_skill_idx(self, global_skill_idx):
        """
        Returns the algorithm and skill vector from the global skill index.
        """
        assert global_skill_idx < len(self.bag_of_skills), f"Global Skill index {global_skill_idx} out of bounds ({len(self.bag_of_skills)})"
        algo_name = self.get_algo_from_skill_idx(global_skill_idx)
        skill_z = self.get_skill_from_skill_idx(global_skill_idx)
        return algo_name, skill_z