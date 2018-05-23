"""
Train/test runner for RL
"""
import numpy as np

from rl_ttt import game
from rl_ttt import utils


class Runner(object):

    def __init__(self, agents, environment):
        self.agents = agents
        self.env = environment
        self.game_results = {
            game.GameStatus.X_WIN: 0,
            game.GameStatus.O_WIN: 0,
            game.GameStatus.DRAW: 0,
        }

    def train(self, nb_steps=None, nb_episodes=None):
        sc = utils.get_stop_condition(nb_steps, nb_episodes)

        step = 0
        episode = 0

        observation = None
        reward = None
        done = None
        info = None

        while sc(step, episode):
            print('[Runner] Step =', step)
            if observation is None:
                observation = self.env.reset()
                self.env.render()
                episode += 1
                print('[Runner] Episode =', episode)

            for agent in self.agents:
                action = agent.forward(observation)
                observation, reward, done, info = self.env.step(action)
                self.env.render()
                if done:
                    break
                assert reward == 0
                agent.backward(reward, terminal=False)

            if done:
                self.game_results[info['status']] += 1

                for agent, reward_multiplier in zip(self.agents, [1, -1]):
                    agent.forward(observation)
                    agent.backward(reward_multiplier * reward, terminal=True)
                observation = None

            step += 1

        self._print_stats()

    def test(self):
        raise NotImplementedError()

    def _print_stats(self):
        x_wins = self.game_results[game.GameStatus.X_WIN]
        o_wins = self.game_results[game.GameStatus.O_WIN]
        draws = self.game_results[game.GameStatus.DRAW]

        total_games = x_wins + o_wins + draws

        fmt_str = "X_WINS\t{}\t{} (%)\n" \
                  "O_WINS\t{}\t{} (%)\n" \
                  "DRAW\t{}\t{} (%)"

        msg = fmt_str.format(
            x_wins, np.round(100 * x_wins / total_games),
            o_wins, np.round(100 * o_wins / total_games),
            draws, np.round(100 * draws / total_games),
        )

        print(msg)
