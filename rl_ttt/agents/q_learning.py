"""
Q-Learning agent implementation
"""
import collections

import numpy as np

from rl_ttt.agents import base


def from_config(cfg, stats):
    return QLearningAgent(cfg.marker_type, stats, cfg.learning_rate,
                          cfg.discount_factor, cfg.eps, cfg.batch_mode)


class QLearningAgent(base.Agent):

    def __init__(self, marker_type, stats, learning_rate=0.01,
                 discount_factor=0.5, eps=0.3, batch_mode=False):
        super(QLearningAgent, self).__init__(marker_type, stats)

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.eps = eps
        self.batch_mode = batch_mode

        self.q = collections.defaultdict(lambda: np.random.uniform())
        self.history = []

    def _forward(self, board, possible_actions):
        s_t = board

        if np.random.uniform() < self.eps:
            a_t = np.random.choice(possible_actions)
        else:
            q_values_for_state = [(a, self.q[(s_t, a)])
                                  for a in possible_actions
                                  if (s_t, a) in self.q.keys()]

            if not q_values_for_state:
                a_t = possible_actions[0]
            else:
                a_t = max(q_values_for_state, key=lambda aq: aq[1])[0]

        _ = self.q[(s_t, a_t)]
        self.history.append((s_t, a_t))

        return a_t

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
                # This happens when a new game was started - only the initial
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

        r_t_1 = np.max([self.q[(s_t_1, a)]
                        for a in base.get_possible_actions(s_t_1)
                        if (s_t_1, a) in self.q.keys()])

        self.q[(s_t, a_t)] += alpha * (r + gamma * r_t_1 - self.q[(s_t, a_t)])

    def load_q_values(self, filename):
        with open(filename, 'r') as f:
            lines = f.read().split('\n')

        for line in lines:
            s_t, a_t, q_value = line.split(';')
            s_t = tuple([x[1:-1] for x in s_t.split(',')])
            a_t = int(a_t)
            q_value = float(q_value)
            self.q[(s_t, a_t)] = q_value

    def save_q_values(self, filename):
        lines = []
        for s_t, a_t in sorted(self.q.keys()):
            s_t = str(s_t)[1:-1].replace(' ', '')
            lines.append(f'{s_t};{a_t};{self.q[(s_t, a_t)]}')

        with open(filename, 'w') as f:
            f.write('\n'.join(lines))
