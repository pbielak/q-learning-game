"""
Module for random game generator
"""
from tqdm import tqdm

from rl_ttt import agents as ttt_agents
from rl_ttt import environment as ttt_env
from rl_ttt.supervised_learning import utils
from rl_ttt import game as ttt_game


class RandomGameGenerator(object):

    def __init__(self):
        self.agents = [ttt_agents.random.RandomAgent('X', stats=None),
                       ttt_agents.random.RandomAgent('O', stats=None)]
        self.env = ttt_env.TicTacToeEnv(game=ttt_game.TicTacToe(),
                                        gui_callback=None)
        self.history = []
        self.pb = None

    def run(self, nb_episodes):
        episode = 0
        observation = None
        reward = None
        done = None

        episode_data = []

        self.pb = tqdm(desc='Episode/Game', total=nb_episodes, position=0)
        while episode < nb_episodes:
            if observation is None:
                self.env.game = utils.random_non_terminal_game()
                observation = self.env.game.board
                episode += 1
                self.pb.update(episode - self.pb.n)

            for agent in self.agents:
                s_t = tuple(observation)
                a_t = agent.forward(observation)

                observation, reward, done, _ = self.env.step(a_t)

                episode_data.append((tuple(utils.board_to_values(s_t)), a_t))

                if done:
                    break

            if done:
                self.history.append((reward, episode_data))
                episode_data = []
                observation = None

        self.pb.close()

    def save(self, filename, append=False):
        history = []

        for entry in self.history:
            reward, episode_data = entry

            episode_data = [(a_t, s_t) for s_t, a_t in episode_data]
            episode_data = ';'.join(map(lambda x: str(x), episode_data))
            history.append(f'{reward};{episode_data}')

        flag = 'w' if not append else 'a'
        with open(filename, flag) as f:
                f.write('\n'.join(history))
