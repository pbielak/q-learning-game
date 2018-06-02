"""
Q-Learning agent implementation
"""
import itertools

import numpy as np

from rl_ttt.agents import base


def from_config(cfg, gui):
    return QLearningAgent(cfg.marker_type, gui, cfg.learning_rate,
                          cfg.discount_factor, cfg.eps, cfg.batch_mode)


class QLearningAgent(base.Agent):

    def __init__(self, marker_type, gui_callback, learning_rate=0.01,
                 discount_factor=0.5, eps=0.3, batch_mode=False):
        super(QLearningAgent, self).__init__(marker_type, gui_callback)

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.eps = eps
        self.batch_mode = batch_mode
        self.q = {}

        self.history = []

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

        self.history.append((board, chosen_action))

        return chosen_action

    def _backward(self, reward, terminal):
        if self.batch_mode:
            if not terminal:
                return

            for idx in range(len(self.history) - 1):
                s_t, a_t = self.history[idx]
                s_t_1, _ = self.history[idx + 1]
                self._update_q_value(s_t, a_t, s_t_1, reward)

        else:
            self.history = self.history[-2:]

            if len(self.history) != 2:
                # This happens when a new game was started (only the initial
                # observation will be present. We need to wait for the next one.
                return

            s_t, a_t = self.history[0]
            s_t_1, _ = self.history[1]

            self._update_q_value(s_t, a_t, s_t_1, reward)

        if terminal:
            self.history = []

    def _update_q_value(self, s_t, a_t, s_t_1, r):
        alpha = self.learning_rate
        gamma = self.discount_factor
        r_t_1 = np.max([self.q[(s_t_1, action)]
                        for action in base.get_possible_actions(s_t_1)])

        self.q[(s_t, a_t)] += alpha * (r + gamma * r_t_1 - self.q[(s_t, a_t)])

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
