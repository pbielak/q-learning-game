"""
Q-Learning agent implementation
"""
import itertools

import numpy as np

from rl_ttt.agents import base


def from_config(cfg, gui):
    return QLearningAgent(cfg.marker_type, gui, cfg.learning_rate,
                          cfg.discount_factor, cfg.eps)


class QLearningAgent(base.Agent):

    def __init__(self, marker_type, gui_callback, learning_rate=0.01,
                 discount_factor=0.5, eps=0.3):
        super(QLearningAgent, self).__init__(marker_type, gui_callback)

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.eps = eps
        self.q = {}

        self._previous_observation = tuple([''] * 9)  # Initially board is empty
        self._current_observation = tuple([''] * 9)
        self._chosen_action = None

        self._init_q_table()

    def _init_q_table(self):
        for state, action in itertools.product(self.all_states,
                                               self.all_actions):
            self.q[(state, action)] = np.random.uniform()

    def _forward(self, board, possible_actions):
        if np.random.uniform() < self.eps:
            chosen_action = np.random.choice(possible_actions)
        else:
            q_values_for_state = [(action, self.q[(board, action)])
                                  for action in possible_actions]
            chosen_action = max(q_values_for_state, key=lambda aq: aq[1])[0]

        self._previous_observation = self._current_observation
        self._current_observation = board
        self._chosen_action = chosen_action

        return chosen_action

    def backward(self, reward, terminal):
        state_action_t = (self._previous_observation, self._chosen_action)
        alpha = self.learning_rate
        gamma = self.discount_factor
        estimated_future_reward = np.max(
            [self.q[(self._current_observation, action)]
             for action in self.all_actions]
        )

        self.q[state_action_t] += alpha * (reward +
                                           gamma * estimated_future_reward
                                           - self.q[state_action_t])

        self.gui_callback(reward, list(self.q.values()))

    def load_q_values(self, filename):
        with open(filename, 'r') as f:
            lines = f.read().split('\n')

        for state_idx, state in enumerate(self.all_states):
            q_values = [float(x) for x in lines[state_idx].split(';')]
            for action_idx, action in enumerate(self.all_actions):
                self.q[(state, action)] = q_values[action_idx]

    def save_q_values(self, filename):
        lines = [
            ';'.join([str(self.q[(state, action)])
                      for action in self.all_actions])
            for state in self.all_states
        ]

        with open(filename, 'w') as f:
            for line in lines:
                f.write(line + '\n')
