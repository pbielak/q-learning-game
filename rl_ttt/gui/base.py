"""
Tic-Tac-Toe GUI
"""


class GUI(object):

    def __init__(self, cfg, stats):
        self.cfg = cfg
        self.stats = stats

    def draw(self):
        pass

    def refresh(self):
        self._refresh_agent_gui()
        self._refresh_game_outcomes()

    def update_env_gui(self, **kwargs):
        pass

    def _refresh_agent_gui(self):
        pass

    def _refresh_game_outcomes(self):
        pass
