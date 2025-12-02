import gymnasium as gym
from gymnasium import spaces

class MetaControllerEnv(gym.Env):
    def __init__(self, base_env, skill_registry, skill_duration=10):
        super().__init__()
        self.base_env = base_env
        self.registry = skill_registry
        self.skill_duration = skill_duration # "How long selected skill should run" 
        
        # Action Space: Select one of the frozen skills
        # self.action_space = spaces.Discrete(self.registry.num_actions)
        self._discrete_actions = [
            self._env.actions.forward,
            self._env.actions.left,
            self._env.actions.right,
            self._env.actions.pickup,
            self._env.actions.drop,
            self._env.actions.toggle,
        ]
        self._num_actions = len(self._discrete_actions)
        self.action_space = akro.Box(
            low=-action_scale,
            high=action_scale,
            shape=(self._num_actions,),
            dtype=np.float32,
        )        

        # Observation Space: Must match what your PPO expects. 
        # Assuming MiniGrid returns an image or a flattened state.
        self.observation_space = self.base_env.observation_space

    def reset(self, seed=None, options=None):
        return self.base_env.reset(seed=seed, options=options)

    def step(self, action_idx):
        """
        The Meta-Step: Execute one skill for k steps.
        """
        total_reward = 0
        terminated = False
        truncated = False
        
        # --- The Scheduler Loop  ---
        for _ in range(self.skill_duration):
            # 1. Get current observation from environment
            # Note: We need the LAST observation to query the policy
            # (In a real implementation, cache 'obs' from the loop start)
            
            # 2. Ask the specific sub-skill for a primitive action
            # We use the current observation (self.last_obs) 
            primitive_action = self.registry.get_action(action_idx, self.last_obs)
            
            # 3. Step the physical environment
            obs, reward, terminated, truncated, info = self.base_env.step(primitive_action)
            self.last_obs = obs # Update for next micro-step
            
            total_reward += reward
            
            # If the task is solved or failed during the skill, stop early
            if terminated or truncated:
                break
                
        # Return the aggregated experience to the Meta-Controller
        return obs, total_reward, terminated, truncated, info

    # Helper to capture the obs for the registry
    def reset(self, **kwargs):
        obs, info = self.base_env.reset(**kwargs)
        self.last_obs = obs
        return obs, info