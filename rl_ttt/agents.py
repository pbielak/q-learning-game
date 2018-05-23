"""
Agent implementations
"""
import itertools

import numpy as np

from rl_ttt.game import FieldStates
from rl_ttt import utils


def process_observation(observation):
    board = tuple([field.value for field in observation])
    possible_actions = [idx for idx, field in enumerate(board)
                        if field == FieldStates.EMPTY_FIELD.value]
    return board, possible_actions


class Agent(object):
    def __init__(self, name, gui_callback):
        self.name = name
        self.gui_callback = gui_callback

    @utils.log_call(log_result=True)
    def forward(self, observation):
        board, possible_actions = process_observation(observation)

        if not possible_actions:
            return -1

        chosen_action = self._forward(board, possible_actions)
        return chosen_action

    def _forward(self, board, possible_actions):
        pass

    @utils.log_call(log_args=True)
    def backward(self, reward, terminal):
        self._backward(reward, terminal)
        self.gui_callback(reward)

    def _backward(self, reward, terminal):
        pass


class QLearningAgent(Agent):

    def __init__(self, name, gui_callback, learning_rate=0.01,
                 discount_factor=0.5):
        super(QLearningAgent, self).__init__(name, gui_callback)

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

    def _forward(self, board, possible_actions):
        q_values_for_state = [(action, self.q[(board, action)])
                              for action in possible_actions]
        chosen_action = max(q_values_for_state, key=lambda aq: aq[1])[0]

        self._previous_observation = self._current_observation
        self._current_observation = board
        self._chosen_action = chosen_action

        return chosen_action

    def _backward(self, reward, terminal):
        state_action_t = (self._previous_observation, self._chosen_action)
        alpha = self.learning_rate
        gamma = self.discount_factor
        estimated_future_reward = np.max(
            [self.q[(self._current_observation, action)]
             for action in list(range(9))]
        )

        self.q[state_action_t] += alpha * (reward +
                                           gamma * estimated_future_reward
                                           - self.q[state_action_t])


class RandomAgent(Agent):

    def __init__(self, name, gui_callback):
        super(RandomAgent, self).__init__(name, gui_callback)

    def _forward(self, board, possible_actions):
        return np.random.choice(possible_actions)

    def _backward(self, reward, terminal):
        pass
