"""
RL environment
"""
from rl_ttt import game as ttt_game


class TicTacToeEnv(object):

    def __init__(self, game, gui_callback):
        self.nb_step = 0
        self.game = game
        self.gui_callback = gui_callback

    def step(self, action):
        self.nb_step += 1
        field_idx = action

        if field_idx >= 0:
            assert self.game.is_field_empty(field_idx)
            self.game.set_field(field_idx, self.game.current_player)

        observation = self.game.board
        reward = self._get_reward()
        done = self.game.is_terminal()
        info = {'status': self.game.status}

        return observation, reward, done, info

    def _get_reward(self):
        rewards = {
            ttt_game.GameStatus.PLAYING: 0,
            ttt_game.GameStatus.DRAW: 0,
            ttt_game.GameStatus.X_WIN: 1,
            ttt_game.GameStatus.O_WIN: -1
        }
        return rewards[self.game.status]

    def reset(self):
        self.game.reset()
        return self.game.board

    def render(self):
        fmt_str = 'Current step: {}\n' \
                  'Game status: {}'

        msg = fmt_str.format(
            self.nb_step,
            self.game.status,
        )

        self.gui_callback(msg=msg, board=self.game.board)
