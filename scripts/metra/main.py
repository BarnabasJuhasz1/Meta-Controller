import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt
import gymnasium as gym

from example_minigrid import SimpleEnv

from metraTester import DiscreteSkillTester, MetraWrapper

from models import ModelManager

from metraDiscrete import DiscreteSkillMetra




def main():


    if torch.cuda.is_available():
        print("CUDA enabled")
        device = torch.device("cuda")
        print(torch.cuda.get_device_name(device))
    else:
        print("CUDA not available")
        device = torch.device("cpu")

    testing = True
    
 
    env = SimpleEnv()
    
    
    nSkills = 8  
    
    mM=ModelManager(phiNum=2,polNum=2)

    metra = DiscreteSkillMetra(env, mM ,n_skills=nSkills,lmbd=30.0, device=device)
    

    metra.train(num_epochs=100, steps_per_epoch=50, log_interval=100)
    
    metra.save_models()
    
    envTest = SimpleEnv(render_mode="rgb_array")

    if testing:
    
        print("Testing All Skills")

        tester = DiscreteSkillTester(envTest, metra.phi,  metra.policy, metra.skill_embeddings, n_skills=nSkills, device=device)
    
        results = tester.test_all_skills(
            render=True,                    
            max_steps=200,
            record_video=True,             
            plot_trajectories=True,         
            plot_latent_space=True,         
            plot_unified_grid=True          
        )
    
    return "done"

if __name__ == "__main__":
    print(main())
