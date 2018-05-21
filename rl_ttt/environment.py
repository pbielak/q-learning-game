"""
RL environment
"""
from rl_ttt import game
from rl_ttt import utils


class TicTacToeEnv(object):

    def __init__(self, gui_callback=utils.default_console_gui_callback):
        self.nb_step = 0
        self.game = game.TicTacToe()
        self.gui_callback = gui_callback

    def step(self, action):
        self.nb_step += 1
        field_idx = action

        if field_idx >= 0:
            if self.game.is_field_empty(field_idx):
                self.game.set_field(field_idx, self.game.current_player)

                if self.game.is_terminal():
                    reward = 1 if self.game.has_won()[1] == game.FieldStates.X_MARKER else -1
                else:
                    reward = 0

            else:
                reward = -1
        else:
            if self.game.is_terminal():
                reward = 1 if self.game.has_won()[1] == game.FieldStates.X_MARKER else -1
            else:
                reward = 0

        # TODO: convert observation to proper format (!)
        observation = (self.game.board,
                       self.game.is_terminal(),
                       self.game.has_won())

        done = self.game.is_terminal()
        info = {}

        return observation, reward, done, info

    def reset(self):
        self.game.reset()
        return (self.game.board,
                self.game.is_terminal(),
                self.game.has_won())

    def render(self):
        fmt_str = 'Current step: {}\n' \
                  'Game: {}\n' \
                  'Current player: {}\n' \
                  'Is terminal: {} | Won:{}'

        msg = fmt_str.format(
            self.nb_step,
            list(map(lambda x: x.value, self.game.board)),
            self.game.current_player.value,
            self.game.is_terminal(),
            self.game.has_won()
        )

        self.gui_callback(msg=msg, board=self.game.board)
