"""
Util functions for SL
"""
import numpy as np

from rl_ttt import game as ttt_game


def board_to_values(board):
    return [f.value for f in board]


def values_to_board(values):
    value_to_field = {
        '': ttt_game.FieldStates.EMPTY_FIELD,
        'X': ttt_game.FieldStates.X_MARKER,
        'O': ttt_game.FieldStates.O_MARKER,
    }

    return [value_to_field[v] for v in values]


def random_board():
    field_types = [ttt_game.FieldStates.EMPTY_FIELD,
                   ttt_game.FieldStates.X_MARKER,
                   ttt_game.FieldStates.O_MARKER]
    return list(np.random.choice(field_types,
                                 ttt_game.TicTacToe.BOARD_SIZE ** 2))


def random_non_terminal_game():
    game = ttt_game.TicTacToe()
    while True:
        game.board = random_board()
        game.round = (ttt_game.TicTacToe.BOARD_SIZE ** 2) \
                     - game.board.count(ttt_game.FieldStates.EMPTY_FIELD)
        if not game.is_terminal():
            break

    return game
