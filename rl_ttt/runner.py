"""
Train/test runner for RL
"""
import numpy as np
from tqdm import tqdm

from rl_ttt import game


class Runner(object):

    def __init__(self, agents, environment):
        self.agents = agents
        self.env = environment
        self.game_results = {
            game.GameStatus.X_WIN: 0,
            game.GameStatus.O_WIN: 0,
            game.GameStatus.DRAW: 0,
        }

    def train(self, nb_episodes=None):
        progress_bar = tqdm(desc='Episode/Game', total=nb_episodes, position=0)

        x_wins_pb = tqdm(desc='X Wins', total=nb_episodes, position=2)
        o_wins_pb = tqdm(desc='O Wins', total=nb_episodes, position=3)
        draws_pb = tqdm(desc='Draws', total=nb_episodes, position=4)

        episode = 0

        observation = None
        reward = None
        done = None
        info = None

        while episode < nb_episodes:
            if observation is None:
                observation = self.env.reset()
                self.env.render()
                episode += 1
                progress_bar.update(1)

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
                if info['status'] == game.GameStatus.X_WIN:
                    x_wins_pb.update(1)
                elif info['status'] == game.GameStatus.O_WIN:
                    o_wins_pb.update(1)
                elif info['status'] == game.GameStatus.DRAW:
                    draws_pb.update(1)

                for agent, reward_multiplier in zip(self.agents, [1, -1]):
                    agent.forward(observation)
                    agent.backward(reward_multiplier * reward, terminal=True)
                observation = None

        progress_bar.close()
        x_wins_pb.close()
        o_wins_pb.close()
        draws_pb.close()
        self._print_stats()

    def test(self):
        raise NotImplementedError()

    def _print_stats(self):
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
