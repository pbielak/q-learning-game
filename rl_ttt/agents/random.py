"""
Random agent implementation
"""
import numpy as np

from rl_ttt.agents import base


def from_config(cfg, gui):
    return RandomAgent(cfg.marker_type, gui)


class RandomAgent(base.Agent):

    def __init__(self, marker_type, gui_callback):
        super(RandomAgent, self).__init__(marker_type, gui_callback)

    def _forward(self, board, possible_actions):
        return np.random.choice(possible_actions)

    def backward(self, reward, terminal):
        self.gui_callback(reward, None)
