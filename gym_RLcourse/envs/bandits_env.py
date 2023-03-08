from typing import Optional
import numpy as np
import gymnasium as gym

class MultiarmedBanditsEnv(gym.Env):
    """Environment for multiarmed bandits"""
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, render_mode: Optional[str]=None, nr_arms=10):
        self.action_space = gym.spaces.Discrete(nr_arms)
        self.observation_space = gym.spaces.Discrete(1)
        self.state = 0


    def step(self, action : int):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"

        reward = self.np_random.standard_normal(1) + self.values[action]

        return self.state, reward.item(), False, False, {'optimal' : self.optimal}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        
        self.values = self.np_random.standard_normal(self.action_space.n)
        self.optimal = np.argmax(self.values)

        return self.state, {'optimal' : self.optimal}

    def render(self, mode='human', close=False):
        return "You are playing a %d-armed bandit" % self.action_space.n
