"""
Module for statistics class
"""
import numpy as np

from rl_ttt import game


class Stats(object):

    def __init__(self):
        self.episode = 0
        self.game_results = {
            game.GameStatus.X_WIN: 0,
            game.GameStatus.O_WIN: 0,
            game.GameStatus.DRAW: 0,
        }

    def print(self):
        x_wins = self.game_results[game.GameStatus.X_WIN]
        o_wins = self.game_results[game.GameStatus.O_WIN]
        draws = self.game_results[game.GameStatus.DRAW]

        total_games = x_wins + o_wins + draws

        fmt_str = "\n\n\n\n\n" \
                  "X_WINS\t{}\t{} (%)\n" \
                  "O_WINS\t{}\t{} (%)\n" \
                  "DRAW\t{}\t{} (%)"

        msg = fmt_str.format(
            x_wins, np.round(100 * x_wins / total_games),
            o_wins, np.round(100 * o_wins / total_games),
            draws, np.round(100 * draws / total_games),
        )

        print(msg)
