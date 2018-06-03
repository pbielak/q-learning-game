"""
Console based GUI
"""
from tqdm import tqdm

from rl_ttt.gui import base


class ConsoleGUI(base.GUI):

    def __init__(self, cfg, stats):
        super(ConsoleGUI, self).__init__(cfg, stats)

        self.episode_pb = None

    def draw(self):
        self.episode_pb = tqdm(desc='Episode/Game',
                               total=self.cfg.nb_episodes,
                               position=0)

    def _refresh_game_outcomes(self):
        self.episode_pb.update(self.stats.episode - self.episode_pb.n)
