"""
Random agent implementation
"""
import numpy as np

from rl_ttt.agents import base


def from_config(cfg, stats):
    return RandomAgent(cfg.marker_type, stats)


class RandomAgent(base.Agent):

    def __init__(self, marker_type, stats):
        super(RandomAgent, self).__init__(marker_type, stats)

    def _forward(self, board, possible_actions):
        return np.random.choice(possible_actions)

    def _backward(self, reward, terminal):
        return
