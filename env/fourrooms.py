import logging
import pygame
import math
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np

logger = logging.getLogger(__name__)

class FourRooms(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'render_fps' : 50
    }

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        #pygame info
        self.cell_size = 40  # Size of each cell in pixels
        self.screen = None
        self.clock = None

        layout = """\
wwwwwwwwwwwww
w     w     w
w     w     w
w           w
w     w     w
w     w     w
ww wwww     w
w     www www
w     w     w
w     w     w
w           w
w     w     w
wwwwwwwwwwwww
"""

        self.occupancy = np.array([list(map(lambda c: 1 if c=='w' else 0, line)) for line in layout.splitlines()])

        # From any state the agent can perform one of four actions, up, down, left or right
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0., high=1., shape=(np.sum(self.occupancy == 0),))

        self.directions = [np.array((-1,0)), np.array((1,0)), np.array((0,-1)), np.array((0,1))]
        self.rng = np.random.RandomState(1234)

        self.tostate = {}
        statenum = 0
        for i in range(13):
            for j in range(13):
                if self.occupancy[i, j] == 0:
                    self.tostate[(i,j)] = statenum
                    statenum += 1
        self.tocell = {v:k for k,v in self.tostate.items()}

        self.goal = 62 # East doorway
        self.init_states = list(range(self.observation_space.shape[0]))
        self.init_states.remove(self.goal)
        self.ep_steps = 0

        # Pygame setup (if rendering is enabled)
        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode(
                (self.occupancy.shape[1] * self.cell_size, self.occupancy.shape[0] * self.cell_size)
            )
            self.clock = pygame.time.Clock()

    def choose_goal(self, goal):
        self.goal = goal
        self.init_states = list(range(self.observation_space.shape[0]))
        self.init_states.remove(self.goal)

    def seed(self, seed=None):
        return self._seed(seed)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def empty_around(self, cell):
        avail = []
        for action in range(self.action_space.n):
            nextcell = tuple(cell + self.directions[action])
            if not self.occupancy[nextcell]:
                avail.append(nextcell)
        return avail

    def reset(self, *, seed=None, options=None):
        state = self.rng.choice(self.init_states)
        self.currentcell = self.tocell[state]
        self.ep_steps = 0
        return self.get_state(state), {}

    # def reset(self):
    #     state = self.rng.choice(self.init_states)
    #     self.currentcell = self.tocell[state]
    #     self.ep_steps = 0
    #     return self.get_state(state), {}

    def switch_goal(self):
        prev_goal = self.goal
        self.goal = self.rng.choice(self.init_states)
        self.init_states.append(prev_goal)
        self.init_states.remove(self.goal)
        assert prev_goal in self.init_states
        assert self.goal not in self.init_states

    def get_state(self, state):
        s = np.zeros(self.observation_space.shape[0])
        s[state] = 1
        return s

    def close(self):
        if self.render_mode == "human" and self.screen is not None:
            pygame.display.quit()  # Close the display
            pygame.quit()  # Quit pygame
            self.screen = None

    def render(self):
        #TODO: add functionality for showing current option?
        if self.render_mode == "human":
            if self.screen is None:
                raise ValueError("Environment is not set up for rendering. Use render_mode='human'.")

            # Handle pygame events to prevent freezing
            for event in pygame.event.get():
                if event.type == pygame.QUIT:  # Handle window close event
                    pygame.quit()
                    exit()

            # Draw the grid
            self.screen.fill((255, 255, 255))  # White background
            for i in range(self.occupancy.shape[0]):
                for j in range(self.occupancy.shape[1]):
                    color = (0, 0, 0) if self.occupancy[i, j] == 1 else (200, 200, 200)  # Walls or empty space
                    pygame.draw.rect(
                        self.screen,
                        color,
                        pygame.Rect(j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size),
                    )

            # Draw the goal
            goal_cell = self.tocell[self.goal]
            pygame.draw.rect(
                self.screen,
                (0, 255, 0),  # Green
                pygame.Rect(goal_cell[1] * self.cell_size, goal_cell[0] * self.cell_size, self.cell_size,
                            self.cell_size),
            )

            # Draw the agent
            pygame.draw.rect(
                self.screen,
                (255, 0, 0),  # Red
                pygame.Rect(
                    self.currentcell[1] * self.cell_size, self.currentcell[0] * self.cell_size, self.cell_size,
                    self.cell_size
                ),
            )

            # Render timestep counter
            if not hasattr(self, "font"):
                pygame.font.init()
                self.font = pygame.font.Font(None, 36)  # Default font, size 36
            timestep_text = self.font.render(f"t = {self.ep_steps}", True, (240, 250, 250))
            self.screen.blit(timestep_text, (10, 10))  # Position at top-left corner

            # Update the display
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])  # Limit FPS

    def step(self, action):
        """
        The agent can perform one of four actions,
        up, down, left or right, which have a stochastic effect. With probability 2/3, the actions
        cause the agent to move one cell in the corresponding direction, and with probability 1/3,
        the agent moves instead in one of the other three directions, each with 1/9 probability. In
        either case, if the movement would take the agent into a wall then the agent remains in the
        same cell.
        We consider a case in which rewards are zero on all state transitions.
        """
        self.ep_steps += 1

        nextcell = tuple(self.currentcell + self.directions[action])
        if not self.occupancy[nextcell]:
            if self.rng.uniform() < 1/3.:
                empty_cells = self.empty_around(self.currentcell)
                self.currentcell = empty_cells[self.rng.randint(len(empty_cells))]
            else:
                self.currentcell = nextcell

        state = self.tostate[self.currentcell]
        done = state == self.goal
        reward = float(done)
        truncated = False

        if not done and self.ep_steps >= 1000:
            truncated = True ; reward = 0.0

        return self.get_state(state), reward, done, truncated, None

class FourRooms_m(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'render_fps' : 50
    }

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        #pygame info
        self.cell_size = 40  # Size of each cell in pixels
        self.screen = None
        self.clock = None


        layout = """\
wwwwwwwwwwwww
w   w       w
w   w       w
w   w   w   w
w   w   w   w
w   w   w   w
w   w   w   w
w   w   w   w
w   w   w   w
w   w   w   w
w       w   w
w       w   w
wwwwwwwwwwwww
"""

        self.occupancy = np.array([list(map(lambda c: 1 if c=='w' else 0, line)) for line in layout.splitlines()])

        # From any state the agent can perform one of four actions, up, down, left or right
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0., high=1., shape=(np.sum(self.occupancy == 0),))

        self.directions = [np.array((-1,0)), np.array((1,0)), np.array((0,-1)), np.array((0,1))]
        self.rng = np.random.RandomState(1234)

        self.tostate = {}
        statenum = 0
        for i in range(13):
            for j in range(13):
                if self.occupancy[i, j] == 0:
                    self.tostate[(i,j)] = statenum
                    statenum += 1
        self.tocell = {v:k for k,v in self.tostate.items()}

        self.goal = 62 # East doorway
        self.init_states = list(range(self.observation_space.shape[0]))
        self.init_states.remove(self.goal)
        self.ep_steps = 0

        # Pygame setup (if rendering is enabled)
        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode(
                (self.occupancy.shape[1] * self.cell_size, self.occupancy.shape[0] * self.cell_size)
            )
            self.clock = pygame.time.Clock()

    def choose_goal(self, goal):
        self.goal = goal
        self.init_states = list(range(self.observation_space.shape[0]))
        self.init_states.remove(self.goal)

    def seed(self, seed=None):
        return self._seed(seed)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def empty_around(self, cell):
        avail = []
        for action in range(self.action_space.n):
            nextcell = tuple(cell + self.directions[action])
            if not self.occupancy[nextcell]:
                avail.append(nextcell)
        return avail

    def reset(self, *, seed=None, options=None):
        state = self.rng.choice(self.init_states)
        self.currentcell = self.tocell[state]
        self.ep_steps = 0
        return self.get_state(state), {}

    # def reset(self):
    #     state = self.rng.choice(self.init_states)
    #     self.currentcell = self.tocell[state]
    #     self.ep_steps = 0
    #     return self.get_state(state), {}

    def switch_goal(self):
        prev_goal = self.goal
        self.goal = self.rng.choice(self.init_states)
        self.init_states.append(prev_goal)
        self.init_states.remove(self.goal)
        assert prev_goal in self.init_states
        assert self.goal not in self.init_states

    def get_state(self, state):
        s = np.zeros(self.observation_space.shape[0])
        s[state] = 1
        return s

    def close(self):
        if self.render_mode == "human" and self.screen is not None:
            pygame.display.quit()  # Close the display
            pygame.quit()  # Quit pygame
            self.screen = None

    def render(self):
        if self.render_mode == "human":
            if self.screen is None:
                raise ValueError("Environment is not set up for rendering. Use render_mode='human'.")

            # Handle pygame events to prevent freezing
            for event in pygame.event.get():
                if event.type == pygame.QUIT:  # Handle window close event
                    pygame.quit()
                    exit()

            # Draw the grid
            self.screen.fill((255, 255, 255))  # White background
            for i in range(self.occupancy.shape[0]):
                for j in range(self.occupancy.shape[1]):
                    color = (0, 0, 0) if self.occupancy[i, j] == 1 else (200, 200, 200)  # Walls or empty space
                    pygame.draw.rect(
                        self.screen,
                        color,
                        pygame.Rect(j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size),
                    )

            # Draw the goal
            goal_cell = self.tocell[self.goal]
            pygame.draw.rect(
                self.screen,
                (0, 255, 0),  # Green
                pygame.Rect(goal_cell[1] * self.cell_size, goal_cell[0] * self.cell_size, self.cell_size,
                            self.cell_size),
            )

            # Draw the agent
            pygame.draw.rect(
                self.screen,
                (255, 0, 0),  # Red
                pygame.Rect(
                    self.currentcell[1] * self.cell_size, self.currentcell[0] * self.cell_size, self.cell_size,
                    self.cell_size
                ),
            )

            # Render timestep counter
            if not hasattr(self, "font"):
                pygame.font.init()
                self.font = pygame.font.Font(None, 36)  # Default font, size 36
            timestep_text = self.font.render(f"t = {self.ep_steps}", True, (240, 250, 250))
            self.screen.blit(timestep_text, (10, 10))  # Position at top-left corner

            # Update the display
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])  # Limit FPS

    def step(self, action):
        """
        The agent can perform one of four actions,
        up, down, left or right, which have a stochastic effect. With probability 2/3, the actions
        cause the agent to move one cell in the corresponding direction, and with probability 1/3,
        the agent moves instead in one of the other three directions, each with 1/9 probability. In
        either case, if the movement would take the agent into a wall then the agent remains in the
        same cell.
        We consider a case in which rewards are zero on all state transitions.
        """
        self.ep_steps += 1

        nextcell = tuple(self.currentcell + self.directions[action])
        if not self.occupancy[nextcell]:
            if self.rng.uniform() < 1/3.:
                empty_cells = self.empty_around(self.currentcell)
                self.currentcell = empty_cells[self.rng.randint(len(empty_cells))]
            else:
                self.currentcell = nextcell

        state = self.tostate[self.currentcell]
        done = state == self.goal
        reward = float(done)
        truncated = False

        if not done and self.ep_steps >= 1000:
            truncated = True ; reward = 0.0

        return self.get_state(state), reward, done, truncated, None



if __name__=="__main__":
    env = FourRooms()
    env.seed(3)