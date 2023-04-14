
import gymnasium as gym
from gymnasium.utils import seeding
import numpy as np
import sys
from gymnasium import Env, spaces

from typing import Optional
from gymnasium.envs.toy_text.utils import categorical_sample


class CorridorEnv(Env):
    """
    Creates an environment for Example 9.1 from Reinforcement Learning: An Introduction
    by Sutton and Barto:
    http://incompleteideas.net/book/RLbook2020.pdf

    Adapted from cliffwalking in openai-gym

    A corridor of n_starting states + 2 terminal states
        [0, 0] : terminal state at the left (reward -1)
        [0, n_starting states + 1] : terminal state at the right (reward +1)
        Each step into a non-terminal state gives 0 reward
        Movement into right and left with max_delta is allowed
    """
    metadata = {
        "render_modes": ["human", "ansi"],
        "render_fps": 4,
    }
    def __init__(self, n_starting_states=1000, max_delta=100):
        self.max_delta = max_delta
        self.shape = (1, n_starting_states+2)
        halfway = self.shape[1] // 2
        self.start_state_index = np.ravel_multi_index((0, halfway), self.shape)

        self.nS = nS = np.prod(self.shape)
        self.nA = nA = 2*self.max_delta # agent can go right or left with a max_delta

        # Calculate transition probabilities and rewards
        P = {}
        for s in range(nS):
            position = np.unravel_index(s, self.shape)
            P[s] = {a: [] for a in range(nA)}
            for a in range(nA):
                if a < self.max_delta:
                    delta = -a-1
                else:
                    delta = a - self.max_delta+1
                P[s][a] = self._calculate_transition_prob(position, [0, delta])
        self.P=P


        # Initial State Distribution
        # always starts (approximately) halfway
        isd = np.zeros(nS)
        isd[self.start_state_index] = 1.0
        self.initial_state_distrib = isd
        self.action_space = spaces.Discrete(nA)
        self.observation_space = spaces.Discrete(nS)

    def _limit_coordinates(self, coord):
        """
        Prevent the agent from falling out of the grid world
        :param coord:
        :return:
        """
        coord[0] = min(coord[0], self.shape[0]-1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1]-1 )
        coord[1] = max(coord[1], 0)
        return coord

    def _calculate_transition_prob(self, current, delta):
        """
        Determine the outcome for an action. Transition Prob is always 1.0.
        :param current: Current position on the grid as (row, col)
        :param delta: Change in position for transition
        :return: (1.0, new_state, reward, done)
        """
        new_position = np.array(current) + np.array(delta)
        new_position = self._limit_coordinates(new_position).astype(int)
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)

        terminal_state_R = (0, self.shape[1]-1)
        terminal_state_L = (0, 0)
        is_done_R = tuple(new_position) == terminal_state_R
        is_done_L = tuple(new_position) == terminal_state_L

        if is_done_R:
            return [(1.0, new_state, 1, is_done_R)]
        if is_done_L:
            return [(1.0, new_state, -1, is_done_L)]
        return [(1.0, new_state, 0, False)]

    def render(self, mode='human'):
        outfile = sys.stdout
        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            if self.s == s:
                output = " x "
            elif position == (0, 0):
                output = " TL"
            elif position == (0, self.shape[1] - 1):
                output = " TR "
            else:
                output = " o "

            if position[1] == 0:
                output = output.lstrip()
            if position[1] == self.shape[1] - 1:
                output = output.rstrip()
                output += '\n'

            outfile.write(output)
        outfile.write('\n')

    def transition_matrix(self):
        return self.P

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