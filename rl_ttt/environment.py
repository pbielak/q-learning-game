"""
RL environment
"""
from rl import core


class TicTacToeEnv(core.Env):

    def __init__(self):
        self.nb_step = 0

    def step(self, action):
        self.nb_step += 1

        # TODO
        observation = (1,)
        reward = 0
        done = False
        info = {}
        return observation, reward, done, info

    def reset(self):
        # TODO
        return self.step(None)

    def render(self, mode='human', close=False):
        print('Current step:', self.nb_step)

    def seed(self, seed=None):
        return

    def configure(self, *args, **kwargs):
        return

    def close(self):
        return
