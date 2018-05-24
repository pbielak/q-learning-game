"""
Agent implementations
"""
import itertools

from rl_ttt import game


def process_observation(observation):
    board = tuple([field.value for field in observation])
    possible_actions = [idx for idx, field in enumerate(board)
                        if field == game.FieldStates.EMPTY_FIELD.value]
    return board, possible_actions


class Agent(object):
    all_states = list(itertools.product(['', 'X', 'O'], repeat=9))
    all_actions = list(range(9))

    def __init__(self, marker_type, gui_callback):
        self.marker_type = marker_type
        self.gui_callback = lambda reward, q_values: gui_callback(reward,
                                                                  q_values,
                                                                  marker_type)

    def forward(self, observation):
        board, possible_actions = process_observation(observation)

        if not possible_actions:
            return -1

        chosen_action = self._forward(board, possible_actions)
        return chosen_action

    def _forward(self, board, possible_actions):
        pass

    def backward(self, reward, terminal):
        pass
