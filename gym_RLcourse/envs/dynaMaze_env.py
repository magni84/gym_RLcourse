import numpy as np
import sys
from gymnasium import Env, spaces
from typing import Optional
from gymnasium.envs.toy_text.utils import categorical_sample

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


class DynaMazeEnv(Env):
    """
    Creates an environment for examples of Chapter 8 from Reinforcement Learning: An Introduction
    by Sutton and Barto:
    http://incompleteideas.net/book/RLbook2020.pdf

    Adapted from cliffwalking in openai-gym

    The board is a 6x9 matrix, with (using Numpy matrix indexing):
        start: Depends on maze_type. See init
        goal:  (0, 8)
        obstacle locations: Depends on maze_type. See initialization of self._obstacle.

    maze_type determines the type of the maze
    Dyna-Maze of Figure 8.2: 'DM'
    Maze-LHS of Figure 8.4: 'BL' (stands for Block Left)
    Maze-RHS of Figure 8.4: 'BR' (stands for Block Right)
    Maze-LHS of Figure 8.5: 'BR' (note that this is the same with RHS of Figure 8.4)
    Maze-RHS of Figure 8.5: 'BN' (stands for Block None)
    BN: defaults to this option if input maze type does not match the other accepted inputs

    Reward is zero on all transitions, except to the one into the goal state,  on which it is 1.
    """
    metadata = {
        "render_modes": ["human", "ansi"],
        "render_fps": 4,
    }
    def __init__(self, maze_type):
        self.shape = (6, 9)
        if maze_type == 'DM':
            self.start_state_position = (2, 0)
        else:
            self.start_state_position = (5, 3)
        self.goal_state_position = (0, 8)
        self.start_state_index = np.ravel_multi_index(self.start_state_position, self.shape)
        self.goal_state_index = np.ravel_multi_index(self.goal_state_position, self.shape)

        self.nS = nS = np.prod(self.shape)
        self.nA = nA = 4

        # Obstacle Locations
        self._obstacle = np.zeros(self.shape, dtype=np.bool)
        if maze_type == 'DM':
            self._obstacle[1:4, 2] = True
            self._obstacle[4, 5] = True
            self._obstacle[0:3, 7] = True
        else:
            self._obstacle[3, 1:8] = True  # same for all other mazes
            if maze_type == 'BL':
                self._obstacle[3, 0] = True
            elif maze_type == 'BR':
                self._obstacle[3, 8] = True

        # Calculate transition probabilities and rewards
        P = {}
        for s in range(nS):
            position = np.unravel_index(s, self.shape)
            P[s] = {a: [] for a in range(nA)}
            P[s][UP] = self._calculate_transition_prob(position, [-1, 0])
            P[s][RIGHT] = self._calculate_transition_prob(position, [0, 1])
            P[s][DOWN] = self._calculate_transition_prob(position, [1, 0])
            P[s][LEFT] = self._calculate_transition_prob(position, [0, -1])
        self.P = P
        # Calculate initial state distribution
        # We always start in the same starting state
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
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def _calculate_transition_prob(self, current_position, delta):
        """
        Determine the outcome for an action. Transition Prob is always 1.0.
        :param current: Current position on the grid as (row, col)
        :param delta: Change in position for transition
        :return: (1.0, new_state, reward, done)
        """
        new_position = np.array(current_position) + np.array(delta)
        new_position = self._limit_coordinates(new_position).astype(int)
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)
        if self._obstacle[tuple(new_position)]:
            current_state = np.ravel_multi_index(tuple(current_position), self.shape)
            return [(1.0, current_state, 0, False)]

        is_done = self.goal_state_index == new_state
        if is_done:
            return [(1.0, new_state, 1, is_done)]
        return [(1.0, new_state, 0, is_done)]

    def render(self, mode='human'):
        outfile = sys.stdout

        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            if self.s == s:
                output = " x "
            elif position == self.goal_state_position:
                output = " G "
            elif self._obstacle[position]:
                output = " W "
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