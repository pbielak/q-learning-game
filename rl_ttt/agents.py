"""
Agent implementations
"""
import numpy as np
from rl import core

from rl_ttt.game import FieldStates


class QLearningAgent(core.Agent):
    compiled = True

    def __init__(self, gui_callback):
        super(QLearningAgent, self).__init__()

        self.gui_callback = gui_callback

    def forward(self, observation):
        # TODO
        from time import sleep
        sleep(1)

        board, _, _ = observation
        empty_idxs = [idx for idx, field in enumerate(board) if field == FieldStates.EMPTY_FIELD]

        if not empty_idxs:
            return -1

        return np.random.choice(empty_idxs)

    def backward(self, reward, terminal):
        # TODO

        fmt_str = 'Received reward: {}\n' \
                  'Terminal: {}'

        if self.gui_callback:
            self.gui_callback(None, fmt_str.format(reward, terminal))
        else:
            print('Received reward:', reward)
            print('Terminal', terminal)
            print('-' * 30)

    def load_weights(self, filepath):
        return

    def save_weights(self, filepath, overwrite=False):
        return

    def compile(self, optimizer, metrics=[]):
        return

    @property
    def layers(self):
        return
