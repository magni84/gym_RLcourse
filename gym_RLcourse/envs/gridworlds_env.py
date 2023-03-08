from contextlib import closing
from typing import Optional
from os import path
from io import StringIO
from gymnasium import utils
import numpy as np
import gymnasium as gym
import pygame

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

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
    ]
}

class GridWorldEnv(gym.Env):
    
    metadata = {
        "render_modes": ["human", "ansi", "rgb_array"],
                "render_fps": 4,
    }

    def __init__(
            self, 
            render_mode: Optional[str] = None,
            map_name="Example 4.1"
    ):
        self.desc = desc = np.asarray(MAPS[map_name], dtype="c")
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = (-1, 0)
        self.map_name = map_name
        self.nA = nA = 4
        self.nS = nS = nrow*ncol

        self.initial_state_distrib = np.array(desc == b"S").astype("float64").ravel()
        self.initial_state_distrib /= self.initial_state_distrib.sum()

        self.observation_space = gym.spaces.Discrete(nS)
        self.action_space = gym.spaces.Discrete(nA)

        self.render_mode = render_mode

        self.P = {s: {a: [] for a in range(nA)} for s in range(nS)}
        

        def to_s(row, col):
            return row*ncol + col

        def inc(row, col, a):
            if a == LEFT:
                col = max(col - 1, 0)
            elif a == DOWN:
                row = min(row + 1, nrow - 1)
            elif a == RIGHT:
                col = min(col + 1, ncol - 1)
            elif a == UP:
                row = max(row - 1, 0)
            return (row, col)

        if b'A' in desc:
            assert b'a' in desc, "Map with A but no a is not possible"
            state_a = np.where(desc.ravel() == b'a')[0][0]
        else:
            state_a = None
    
        if b'B' in desc:
            assert b'b' in desc, "Map with B but no b is not possible"
            state_b = np.where(desc.ravel() == b'b')[0][0]
        else:
            state_b = None

        if map_name == "Example 3.5":
            step_reward = 0
            outside_reward = -1
        else:
            step_reward = -1
            outside_reward = -1
        
        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)

                for a in range(4):
                    li = self.P[s][a]
                    letter = desc[row, col]

                    if letter == b'G':
                        li.append( (1.0, s, 0, True) )
                    else:
                        newrow, newcol = inc(row, col, a)
                        newstate = to_s(newrow, newcol)
                        newletter = desc[newrow, newcol]

                        if letter == b'A':
                            newstate = state_a
                            reward = 10.0
                        elif letter == b'B':
                            newstate = state_b
                            reward = 5.0
                        elif newstate == s:
                            reward = outside_reward
                        else:
                            reward = step_reward

                        if letter == b'2':
                            newrow = max(newrow-2, 0)
                            newstate = to_s(newrow, newcol)
                            newletter = desc[newrow, newcol]
                        elif letter == b'1':
                            newrow = max(newrow-1, 0)
                            newstate = to_s(newrow, newcol)
                            newletter = desc[newrow, newcol]

                        terminated = bytes(newletter) in b"G"

                        li.append( (1.0, newstate, reward, terminated) )


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
        self.smallwind = None
        self.largewind = None
        self.lastaction = None


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
            outfile.write(f"  ({['Left', 'Down', 'Right', 'Up'][self.lastaction]})\n")
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



        desc = self.desc.tolist()
        assert isinstance(desc, list), f"desc should be a list or an array, got {desc}"

        self.window_surface.fill((0,0,0))

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
















