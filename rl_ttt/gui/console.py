"""
Console based GUI
"""
from tqdm import tqdm

from rl_ttt.gui import base
from rl_ttt import game


class ConsoleGUI(base.GUI):

    def __init__(self, cfg):
        super(ConsoleGUI, self).__init__(cfg)

        self.episode_pb = tqdm(desc='Episode/Game',
                               total=cfg.nb_episodes, position=0)
        self.x_wins_pb = tqdm(desc='X Wins', total=cfg.nb_episodes, position=2)
        self.o_wins_pb = tqdm(desc='O Wins', total=cfg.nb_episodes, position=3)
        self.draws_pb = tqdm(desc='Draws', total=cfg.nb_episodes, position=4)

    def update_stats(self, stats):
        self.episode_pb.update(stats.episode - self.episode_pb.n)
        self.x_wins_pb.update(
            stats.game_results[game.GameStatus.X_WIN] - self.x_wins_pb.n
        )
        self.o_wins_pb.update(
            stats.game_results[game.GameStatus.O_WIN] - self.o_wins_pb.n
        )
        self.draws_pb.update(
            stats.game_results[game.GameStatus.DRAW] - self.draws_pb.n
        )
