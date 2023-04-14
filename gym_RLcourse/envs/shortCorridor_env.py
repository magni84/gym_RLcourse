import numpy as np
import sys
from gymnasium import Env, spaces
from typing import Optional
from gymnasium.envs.toy_text.utils import categorical_sample

"""
 
 Creates an environment for Example 13.1 from Reinforcement Learning: An Introduction
 by Sutton and Barto:
 http://incompleteideas.net/book/RLbook2020.pdf

 Adapted from cliffwalking in openai-gym

 """


class ShortCorridorEnv(Env):
    metadata = {
        "render_modes": ["human", "ansi"],
        "render_fps": 4,
    }
    
    def __init__(self):  # __init__(self, n_starting_states =100,max_delta=10):
        self.start_state_index = 0
        self.shape = (1, 4)
        self.nS = nS = np.prod(self.shape)
        self.nA = nA = 2

        # Calculate transition probabilities and rewards
        P = {}
        for s in range(nS):
            P[s] = {a: [] for a in range(nA)}
            for a in range(nA):
                P[s][a] = self._calculate_transition_prob(s, a)

        isd = np.zeros(nS)
        isd[self.start_state_index] = 1.0
        self.initial_state_distrib = isd
        self.action_space = spaces.Discrete(nA)
        self.observation_space = spaces.Discrete(nS)


    def _calculate_transition_prob(self, state, action):
        """
        Determine the outcome for an action. Transition Prob is always 1.0.
        :param current: Current position on the grid as (row, col)
        :param delta: Change in position for transition
        :return: (1.0, new_state, reward, done)
        """
        if state != 1:
            new_state = state-np.power(-1,action) # action 0 left, action 1 right
            new_state=max(0,new_state)
            new_state = min(self.shape[1]-1, new_state)
        else: # reverse for state 1
            new_state = state + np.power(-1, action)

        terminal_state_R = self.shape[1]-1
        is_done = new_state == terminal_state_R

        if is_done:
            return [(1.0, new_state, -1, is_done)]

        return [(1.0, new_state, -1, False)]

    def render(self, mode='human'):
        outfile = sys.stdout

        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            if self.s == s:
                output = " x "
            elif position == (0, self.shape[1] - 1):
                output = " G "
            else:
                output = " o "

            if position[1] == 0:
                output = output.lstrip()
            if position[1] == self.shape[1] - 1:
                output = output.rstrip()
                output += '\n'

            outfile.write(output)
        outfile.write('\n')

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.s = categorical_sample(self.initial_state_distrib, self.np_random)
        self.lastaction = None

        if self.render_mode == "human":
            self.render()
        return int(self.s), {"prob": 1}

    def step(self, a):
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, t = transitions[i]
        self.s = s
        self.lastaction = a

        if self.render_mode == "human":
            self.render()
        return (int(s), r, t, False, {"prob": p})