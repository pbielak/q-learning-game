"""
RL environment
"""
import numpy as np

from rl_ttt import game as ttt_game


class TicTacToeEnv(object):

    def __init__(self, game, gui_callback, reward_model):
        self.nb_step = 0
        self.game = game
        self.gui_callback = gui_callback
        self.reward_model = self._load_reward_model(reward_model)

    def step(self, action):
        self.nb_step += 1
        field_idx = action

        prev_observation  = self.game.board

        if field_idx >= 0:
            assert self.game.is_field_empty(field_idx)
            self.game.set_field(field_idx, self.game.current_player)

        observation = self.game.board
        reward = self.reward_model(prev_observation, action)
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

    def _load_reward_model(self, model_type):
        if model_type == 'simple':
            return lambda s, a: self._get_reward()
        elif model_type == 'supervised':
            from keras.models import load_model
            model = load_model('data/nn_model.save')

            return lambda s, a: model.predict(self._transform_to_nn_input(s, a))
        else:
            raise RuntimeError('Unknown reward model!')

    def _transform_to_nn_input(self, state, action):
        marker_mapping = {
            ttt_game.FieldStates.EMPTY_FIELD: [1, 0, 0],
            ttt_game.FieldStates.X_MARKER: [0, 1, 0],
            ttt_game.FieldStates.O_MARKER: [0, 0, 1],
        }

        action_mapping = np.eye(9)

        state = [sp for s in state for sp in marker_mapping[s]]
        action = [int(x) for x in action_mapping[action]]

        nn_input = np.array([np.array(list([*state, *action])).reshape(36), ])
        return nn_input

    def reset(self):
        self.game.reset()
        return self.game.board

    def render(self):
        msg = f'Game status: {self.game.status}'
        self.gui_callback(msg=msg, board=self.game.board)
