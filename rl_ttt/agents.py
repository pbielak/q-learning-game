"""
Agent implementations
"""
import numpy as np

from rl_ttt.game import FieldStates
from rl_ttt import utils


class QLearningAgent(object):

    def __init__(self, name, gui_callback=utils.default_console_gui_callback):
        self.name = name
        self.gui_callback = gui_callback

    def forward(self, observation):
        # TODO
        # from time import sleep
        # sleep(1)

        board, _, _ = observation
        empty_idxs = [idx for idx, field in enumerate(board) if field == FieldStates.EMPTY_FIELD]

        if not empty_idxs:
            print(self.name, 'forward: no action available!')
            return -1

        chosen_action = np.random.choice(empty_idxs)
        print(self.name, 'forward chosen:', chosen_action)
        return chosen_action

    def backward(self, reward, terminal):
        print(self.name, 'backward (', reward, terminal, ')')
        # TODO

        fmt_str = 'Received reward: {} Terminal: {}'

        self.gui_callback(msg=fmt_str.format(reward, terminal),
                          weights=None, reward=reward)
