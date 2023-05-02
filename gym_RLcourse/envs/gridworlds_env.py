from contextlib import closing
from typing import Optional
from os import path
from io import StringIO
from gymnasium import utils
import numpy as np
import gymnasium as gym
import pygame

WEST = 0
SOUTH = 1
EAST = 2
NORTH = 3
NW = 4
NE = 5
SW = 6
SE = 7

MAPS = {
    "Example 4.1": [
        "GSSS",
        "SSSS",
        "SSSS",
        "SSSG"
    ],
    "Example 3.5": [
        "SASBS",
        "SSSSS",
        "SSSbS",
        "SSSSS",
        "SaSSS"
    ],
    "Example 6.5": [ 
        "0001112210",
        "0001112210",
        "0001112210",
        "S001112G10",
        "0001112210",
        "0001112210",
        "0001112210",
    ],
    "Figure 8.2": [
        "0000000WG",
        "00W0000W0",
        "S0W0000W0",
        "00W000000",
        "00000W000",
        "000000000",
    ],
    "Figure 8.4a": [
        "00000000G",
        "000000000",
        "000000000",
        "WWWWWWWW0",
        "000000000",
        "000S00000",
    ],
    "Figure 8.4b": [
        "00000000G",
        "000000000",
        "000000000",
        "0WWWWWWWW",
        "000000000",
        "000S00000",
    ],
    "Figure 8.5a": [
        "00000000G",
        "000000000",
        "000000000",
        "0WWWWWWWW",
        "000000000",
        "000S00000",
    ],
    "Figure 8.5b": [
        "00000000G",
        "000000000",
        "000000000",
        "0WWWWWWW0",
        "000000000",
        "000S00000",
    ], 
    "Example 13.1": [
        "SR0G"
    ]
}


class GridWorldEnv(gym.Env):
    """
        Base environment for different GridWorlds that implement examples from
        Reinforcement Learning: An Introduction by Sutton and Barto
        http://incompleteideas.net/book/RLbook2020.pdf

        Adapted from FrozenLake in Gymnasium:
        https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/toy_text/frozen_lake.py

        MAPS:
        The base class only do the rendering part, and specialized classes
        computes transition probabilities.
        The interpretation of symbols in the maps are

        S - possible starting position
        G - Goal states (show an "ice cream")
        A, a, B, b - From A/B always jump to a/b (see GridWorldABEnv)
        1,2 - Windy grids (see GridWorldWindyEnv)
        W - A wall (shows a "fence")
        Other - Show ground (grass)

        TRANSITIONS:
        The base class implements simple deterministic transitions, always
        moving in the direction of the action, and stays if this takes it
        out of bounds.
        The reward is -1 for each step until reaching a goal state.
    """
    metadata = {
        "render_modes": ["human", "ansi", "rgb_array"],
        "render_fps": 4,
    }

    def __init__(
            self,
            render_mode: Optional[str] = None,
            map_name="Example 4.1",
            nA = 4
    ):
        self.desc = desc = np.asarray(MAPS[map_name], dtype="c")
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = (-1, 0)
        self.map_name = map_name
        self.nA = nA
        self.nS = nS = nrow*ncol

        self.initial_state_distrib = np.array(desc == b"S").astype("float64").ravel()
        self.initial_state_distrib /= self.initial_state_distrib.sum()

        self.observation_space = gym.spaces.Discrete(nS)
        self.action_space = gym.spaces.Discrete(nA)

        self._calculate_transition_probs()

        self.render_mode = render_mode

        # For render_mode = "human"
        self.window_size = (min(64 * ncol, 512), min(64*nrow, 512))
        self.cell_size = (
            self.window_size[0] // self.ncol,
            self.window_size[1] // self.nrow,
        )
        self.window_surface = None
        self.clock = None
        self.ground = None
        self.player = None
        self.goal = None
        self.A = None
        self.a = None
        self.B = None
        self.b = None
        self.R = None
        self.smallwind = None
        self.largewind = None
        self.wall = None
        self.lastaction = None

    def _inc(self, row, col, a):
        """Computes new row and col if we move according to action"""
        if a == WEST:
            col = max(col - 1, 0)
        elif a == SOUTH:
            row = min(row + 1, self.nrow - 1)
        elif a == EAST:
            col = min(col + 1, self.ncol - 1)
        elif a == NORTH:
            row = max(row - 1, 0)
        elif a == NW:
            row = max(row-1, 0)
            col = max(col-1, 0)
        elif a == NE:
            row = max(row-1, 0)
            col = min(col+1, self.ncol-1)
        elif a == SW:
            row = min(row+1, self.nrow-1)
            col = max(col-1, 0)
        elif a == SE:
            row = min(row+1, self.nrow-1)
            col = min(col+1, self.ncol-1)

        return (row, col)

    def _to_s(self, row, col):
        """Convert row and column to state"""
        return row*self.ncol + col

    def _calculate_transition_probs(self): 

        self.P = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}

        for row in range(self.nrow):
            for col in range(self.ncol):
                s = self._to_s(row, col)

                for a in range(self.nA):
                    li = self.P[s][a]
                    letter = self.desc[row][col]

                    if letter == b'G':
                        li.append((1.0, s, 0, True)) 
                    else:
                        newrow, newcol = self._inc(row, col, a)
                        newstate = self._to_s(newrow, newcol)
                        newletter = self.desc[newrow, newcol]

                        terminated = bytes(newletter) in b"G"
                        
                        li.append((1.0, newstate, -1, terminated))

    def step(self, a):
        p, s, r, t = self.P[self.s][a][0]

        self.lastaction = a
        self.s = s

        if self.render_mode == "human":
            self.render()

        return (int(s), r, t, False, {"prob": p})

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):

        super().reset(seed=seed)

        self.s = self.np_random.choice(self.nS, p = self.initial_state_distrib.ravel())

        if self.render_mode == "human":
            self.render()

        return int(self.s), {"prob": 1}

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        if self.render_mode == "ansi":
            return self._render_text()
        else:
            return self._render_gui(self.render_mode)

    def _render_text(self):
        desc = self.desc.tolist()
        outfile = StringIO()

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = [[c.decode("utf-8") for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write(f"  ({['West', 'South', 'East', 'North', 'North west', 'North east', 'South west', 'South east'][self.lastaction]})\n")
        else:
            outfile.write("\n")
        outfile.write("\n".join("".join(line) for line in desc) + "\n")

        with closing(outfile):
            return outfile.getvalue()

    def _render_gui(self, mode):

        if self.window_surface is None:
            pygame.init()

            if mode == "human":
                pygame.display.init()
                pygame.display.set_caption("GridWorld: %s" % (self.map_name))
                self.window_surface = pygame.display.set_mode(self.window_size)

            elif mode == "rgb_array":
                self.window_surface = pygame.Surface(self.window_size)
                

        assert (
            self.window_surface is not None
        ), "Something went wrong with pygame. This should never happen."

        if self.clock is None:
            self.clock = pygame.time.Clock()

        if self.ground is None:
            file = path.join(path.dirname(__file__), "img/ground.png")
            self.ground = pygame.transform.scale(
                pygame.image.load(file), self.cell_size
            )
        if self.player is None:
            file = path.join(path.dirname(__file__), "img/player.png")
            self.player = pygame.transform.scale(
                pygame.image.load(file), self.cell_size
            )
        if self.goal is None:
            file = path.join(path.dirname(__file__), "img/goal.png")
            self.goal = pygame.transform.scale(
                pygame.image.load(file), self.cell_size
            )
        if self.A is None:
            file = path.join(path.dirname(__file__), "img/AA.png")
            self.A = pygame.transform.scale(
                pygame.image.load(file), self.cell_size
            )
        if self.a is None:
            file = path.join(path.dirname(__file__), "img/a.png")
            self.a = pygame.transform.scale(
                pygame.image.load(file), self.cell_size
            )
        if self.B is None:
            file = path.join(path.dirname(__file__), "img/BB.png")
            self.B = pygame.transform.scale(
                pygame.image.load(file), self.cell_size
            )
        if self.b is None:
            file = path.join(path.dirname(__file__), "img/b.png")
            self.b = pygame.transform.scale(
                pygame.image.load(file), self.cell_size
            ) 
        if self.smallwind is None:
            file = path.join(path.dirname(__file__), "img/smallwind.png")
            self.smallwind = pygame.transform.scale(
                pygame.image.load(file), self.cell_size
            )
        if self.largewind is None:
            file = path.join(path.dirname(__file__), "img/largewind.png")
            self.largewind = pygame.transform.scale(
                pygame.image.load(file), self.cell_size
            ) 
        if self.wall is None:
            file = path.join(path.dirname(__file__), "img/fence.png")
            self.wall = pygame.transform.scale(
                pygame.image.load(file), self.cell_size
            )
        if self.R is None:
            file = path.join(path.dirname(__file__), "img/R.png")
            self.R = pygame.transform.scale(
                pygame.image.load(file), self.cell_size
            )

        desc = self.desc.tolist()
        assert isinstance(desc, list), f"desc should be a list or an array, got {desc}"

        self.window_surface.fill((0, 0, 0))

        for y in range(self.nrow):
            for x in range(self.ncol):
                pos = (x * self.cell_size[0], y * self.cell_size[1])
                rect = (*pos, *self.cell_size)

                self.window_surface.blit(self.ground, pos)

                if desc[y][x] == b"G":
                    self.window_surface.blit(self.goal, pos)
                if desc[y][x] == b"A":
                    self.window_surface.blit(self.A, pos)
                if desc[y][x] == b"a":
                    self.window_surface.blit(self.a, pos) 
                if desc[y][x] == b"B":
                    self.window_surface.blit(self.B, pos)
                if desc[y][x] == b"b":
                    self.window_surface.blit(self.b, pos)
                if desc[y][x] == b"1":
                    self.window_surface.blit(self.smallwind, pos)
                if desc[y][x] == b"2":
                    self.window_surface.blit(self.largewind, pos)
                if desc[y][x] == b"W":
                    self.window_surface.blit(self.wall, pos)
                if desc[y][x] == b"R":
                    self.window_surface.blit(self.R, pos)

                pygame.draw.rect(self.window_surface, (180, 200, 230), rect, 1)


        bot_row, bot_col = self.s // self.ncol, self.s % self.ncol
        cell_rect = (bot_col * self.cell_size[0], bot_row * self.cell_size[1])
        self.window_surface.blit(self.player, cell_rect)

        if mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        elif mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window_surface)), axes=(1, 0, 2)
            )


    def close(self):

        if self.window_surface is not None:
            pygame.display.quit()
            pygame.quit()

class GridWorldCorridorEnv(GridWorldEnv):
    """Implements Example 13.1 in Sutton and Barto."""

    def __init__(
        self,
        render_mode: Optional[str] = None, 
        map_name="Example 13.1",
        nA = 2
    ):
        super().__init__(render_mode, map_name, 2)

    def _calculate_transition_probs(self): 

        LEFT = 0
        RIGHT = 1

        self.P = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}

        for s in range(self.ncol):
            row = 0

            for a in range(self.nA):
                li = self.P[s][a]
                letter = self.desc[row][s]

                if letter == b'R':
                    if a == LEFT:
                        a = RIGHT
                    else:
                        a = LEFT

                if letter == b'G':
                    li.append((1.0, s, 0, True)) 
                else:
                    if a == LEFT:
                        newstate = max(s-1, 0)
                    else:
                        newstate = min(s+1, self.ncol-1)

                    newletter = self.desc[row, newstate]

                    if newletter == b"G":
                        terminated = True 
                        reward = 1
                    terminated = bytes(newletter) in b"G"
                    
                    li.append((1.0, newstate, -1, terminated))
        


class GridWorldABEnv(GridWorldEnv):
    """Implements Example 3.5 in Sutton an Barto. 

    TRANSITIONS:
    Moves deterministically the direction of the action except when in A or B
    in which case the state teleports to a and b respectively. 

    Reward 0 for each step, -1 for going out of bounds, +10 leaving A and +5 leaving B. 
    """
    def __init__(
        self, 
        render_mode: Optional[str] = None,
        map_name="Example 3.5",
        nA = 4
    ):
        super().__init__(render_mode, map_name, nA)

    def _calculate_transition_probs(self): 
        if b'A' in self.desc:
            assert b'a' in self.desc, "Map with A but no a is not possible"
            state_a = np.where(self.desc.ravel() == b'a')[0][0]
        else:
            state_a = None
    
        if b'B' in self.desc:
            assert b'b' in self.desc, "Map with B but no b is not possible"
            state_b = np.where(self.desc.ravel() == b'b')[0][0]
        else:
            state_b = None

        self.P = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}

        for row in range(self.nrow):
            for col in range(self.ncol):
                s = self._to_s(row, col)

                for a in range(self.nA):
                    li = self.P[s][a]
                    letter = self.desc[row][col]

                    if letter == b'G':
                        li.append((1.0, s, 0, True)) 
                    else:
                        newrow, newcol = self._inc(row, col, a)
                        newstate = self._to_s(newrow, newcol)
                        newletter = self.desc[newrow, newcol]

                        if letter == b'A':
                            newstate = state_a
                            reward = 10
                        elif letter == b'B':
                            newstate = state_b
                            reward = 5
                        elif newstate == s:
                            reward = -1
                        else:
                            reward = 0

                        terminated = bytes(newletter) in b"G"
                        
                        li.append((1.0, newstate, reward, terminated))


class GridWorldWindyEnv(GridWorldEnv):
    """Implements Example 6.5 in Sutton an Barto. 

    TRANSITIONS:
    Moves deterministically the direction of the action. If it is a windy state (1 or 2) 
    will move up corresponding number of steps. 

    Reward -1 for each step unless goal is reached.
    """
    def __init__(
        self, 
        render_mode: Optional[str] = None,
        map_name="Example 6.5",
        nA = 4
    ):
        super().__init__(render_mode, map_name, nA)

    def _calculate_transition_probs(self): 

        self.P = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}

        for row in range(self.nrow):
            for col in range(self.ncol):
                s = self._to_s(row, col)

                for a in range(self.nA):
                    li = self.P[s][a]
                    letter = self.desc[row][col]

                    if letter == b'G':
                        li.append((1.0, s, 0, True)) 
                    else:
                        newrow, newcol = self._inc(row, col, a)
                        newstate = self._to_s(newrow, newcol)
                        newletter = self.desc[newrow, newcol]

                        if letter == b'2':
                            newrow = max(newrow-2, 0)
                            newstate = self._to_s(newrow, newcol)
                            newletter = self.desc[newrow, newcol]
                        elif letter == b'1':
                            newrow = max(newrow-1, 0)
                            newstate = self._to_s(newrow, newcol)
                            newletter = self.desc[newrow, newcol]

                        terminated = bytes(newletter) in b"G"
                        
                        li.append((1.0, newstate, -1, terminated))


class GridWorldMazeEnv(GridWorldEnv):
    """Implements mazed from Chapter 8 in Sutton an Barto. 

    TRANSITIONS:
    Move deterministically according to action, but stay if moving into 
    fence or out of bounds. 

    Reward is 0 for all transistions except if the goal state is reached in 
    which case the reward is 1. 
    """
    def __init__(
        self, 
        render_mode: Optional[str] = None,
        map_name="Figure 8.2",
        nA = 4
    ):
        super().__init__(render_mode, map_name, nA)

    def _calculate_transition_probs(self): 

        self.P = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}

        for row in range(self.nrow):
            for col in range(self.ncol):
                s = self._to_s(row, col)

                for a in range(self.nA):
                    li = self.P[s][a]
                    letter = self.desc[row][col]

                    if letter == b'G':
                        li.append((1.0, s, 0, True)) 
                    else:
                        newrow, newcol = self._inc(row, col, a)
                        newstate = self._to_s(newrow, newcol)
                        newletter = self.desc[newrow, newcol]

                        if newletter == b'W':
                            newstate = s
                            newletter = letter

                        if newletter == b"G":
                            reward = 1
                            terminated = True
                        else:
                            reward = 0
                            terminated = False
                        
                        li.append((1.0, newstate, reward, terminated))


















