"""
Tic-Tac-Toe GUI
"""


class GUI(object):

    def __init__(self, cfg):
        self.cfg = cfg

    def draw(self):
        pass

    def update_env_gui(self, **kwargs):
        pass

    def update_agent_gui(self, reward, q_values, agent_name):
        pass

    def update_stats(self, stats):
        pass
