import sys
import os
import torch
import numpy as np

# Add the project root to the path so we can import the module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from contoller.meta_controller import PPO

def test_ppo_initialization():
    print("Testing PPO Initialization...")
    state_dim = 4
    action_dim = 2
    lr_actor = 0.0003
    lr_critic = 0.001
    gamma = 0.99
    K_epochs = 4
    eps_clip = 0.2
    has_continuous_action_space = True
    action_std_init = 0.6

    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std_init)
    print("PPO Initialized successfully.")
    return ppo_agent

def test_select_action(ppo_agent):
    print("Testing Select Action...")
    state = np.random.rand(4)
    action = ppo_agent.select_action(state)
    print(f"Action selected: {action}")
    assert action.shape == (2,), "Action shape mismatch for continuous space"
    
    # Test discrete
    ppo_agent_discrete = PPO(4, 2, 0.0003, 0.001, 0.99, 4, 0.2, False, 0.6)
    action_discrete = ppo_agent_discrete.select_action(state)
    print(f"Discrete Action selected: {action_discrete}")
    assert isinstance(action_discrete, int) or isinstance(action_discrete, np.int64), "Action type mismatch for discrete space"

def test_update(ppo_agent):
    print("Testing Update...")
    ppo_agent.buffer.clear() # Clear buffer from previous tests
    # Simulate a few steps
    for _ in range(10):
        state = np.random.rand(4)
        action = ppo_agent.select_action(state)
        reward = np.random.rand()
        is_terminal = False
        
        ppo_agent.buffer.rewards.append(reward)
        ppo_agent.buffer.is_terminals.append(is_terminal)
    
    ppo_agent.update()
    print("Update completed successfully.")

if __name__ == "__main__":
    try:
        agent = test_ppo_initialization()
        test_select_action(agent)
        test_update(agent)
        print("All tests passed!")
    except Exception as e:
        print(f"Test failed with error: {e}")
        sys.exit(1)
