"""
Agent implementations
"""
import itertools

from rl_ttt import game


def to_board(observation):
    return tuple([field.value for field in observation])


def get_possible_actions(board):
    return [idx for idx, field in enumerate(board)
            if field == game.FieldStates.EMPTY_FIELD.value]


def process_observation(observation):
    board = to_board(observation)
    possible_actions = get_possible_actions(board)
    return board, possible_actions


class Agent(object):
    all_states = list(itertools.product(['', 'X', 'O'], repeat=9))
    all_actions = list(range(9))

    def __init__(self, marker_type, stats):
        self.marker_type = marker_type
        self.stats = stats

    def forward(self, observation):
        board, possible_actions = process_observation(observation)

        if not possible_actions:
            return -1

        chosen_action = self._forward(board, possible_actions)
        return chosen_action

    def _forward(self, board, possible_actions):
        pass

    def backward(self, reward, terminal):
        self._backward(reward, terminal)

        if terminal:
            q_values = list(self.q.values()) if hasattr(self, 'q') else None
            self.stats.add_mean_q(self.marker_type, q_values)
            self.stats.add_reward(self.marker_type, reward)

    def _backward(self, reward, terminal):
        pass
