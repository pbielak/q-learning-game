"""
Agent implementations
"""
import itertools

import numpy as np

from rl_ttt.game import FieldStates
from rl_ttt import utils


class Agent(object):
    def __init__(self, name, gui_callback):
        self.name = name
        self.gui_callback = gui_callback

    def forward(self, observation):
        pass

    def backward(self, reward, terminal):
        pass


class QLearningAgent(object):

    def __init__(self, name, gui_callback, learning_rate=0.01,
                 discount_factor=0.5):
        self.name = name
        self.gui_callback = gui_callback
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q = {}

        self._previous_observation = tuple([''] * 9)  # Initially board is empty
        self._current_observation = tuple([''] * 9)
        self._chosen_action = None

        self._init_q_table()

    def _init_q_table(self):
        all_states = list(itertools.product(['X', 'O', ''], repeat=9))
        all_actions = list(range(9))

        for state, action in itertools.product(all_states, all_actions):
            self.q[(state, action)] = np.random.uniform()

    @utils.log_call(log_result=True)
    def forward(self, observation):
        board = tuple([field.value for field in observation])
        empty_idxs = [idx for idx, field in enumerate(board) if field == '']

        if not empty_idxs:
            return -1

        q_vals_for_state = [(action, self.q[(board, action)])
                            for action in empty_idxs]
        chosen_action = max(q_vals_for_state, key=lambda aq: aq[1])[0]

        self._previous_observation = self._current_observation
        self._current_observation = board
        self._chosen_action = chosen_action

        return chosen_action

    @utils.log_call(log_args=True)
    def backward(self, reward, terminal):
        state_action_t = (self._previous_observation, self._chosen_action)
        alpha = self.learning_rate
        gamma = self.discount_factor
        estimated_future_reward = np.max(
            [self.q[(self._current_observation, action)] for action in list(range(9))]
        )

        self.q[state_action_t] += alpha * (reward + gamma * estimated_future_reward - self.q[state_action_t])

        fmt_str = 'Received reward: {} Terminal: {}'

        # weights = []
        # all_states = sorted(itertools.product(['X', 'O', ''], repeat=9))
        # for action in range(9):
        #     action_weights = []
        #     for state in all_states:
        #         action_weights.append(self.q[(state, action)])
        #     weights.append(action_weights)

        self.gui_callback(msg=fmt_str.format(reward, terminal),
                          weights=None, reward=reward)


class RandomAgent(object):

    def __init__(self, name, gui_callback):
        self.name = name
        self.gui_callback = gui_callback

    @utils.log_call(log_result=True)
    def forward(self, observation):
        board = observation
        empty_idxs = [idx for idx, field in enumerate(board)
                      if field == FieldStates.EMPTY_FIELD]

        if not empty_idxs:
            return -1

        chosen_action = np.random.choice(empty_idxs)
        return chosen_action

    @utils.log_call(log_args=True)
    def backward(self, reward, terminal):
        fmt_str = 'Received reward: {} Terminal: {}'

        self.gui_callback(msg=fmt_str.format(reward, terminal),
                          weights=None, reward=reward)
