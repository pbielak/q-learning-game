"""
Tic-Tac-Toe GUI
"""
import matplotlib.pyplot as plt


class GUI(object):
    def update_env_gui(self, **kwargs):
        pass

    def update_agent_gui(self, **kwargs):
        pass


class TicTacToeGUI(GUI):
    def __init__(self):
        self.board_ax = None
        self.perf_ax = None
        self._init_gui()

    def _init_gui(self):
        plt.ion()
        fig = plt.figure(figsize=(14, 7))
        self.board_ax = fig.add_subplot(121)
        self.perf_ax = fig.add_subplot(122)
        plt.show()

    def update_env_gui(self, board, msg):
        ax = self.board_ax

        ax.cla()
        ax.set_title(msg)

        ax.plot(0, 0)
        ax.set_ylim((-15, 15))
        ax.set_xlim((-15, 15))

        ax.axvline(-5)
        ax.axvline(5)
        ax.axhline(5)
        ax.axhline(-5)

        positions = [(-15, 5), (-5, 5), (5, 5),
                     (-15, -5), (-5, -5), (5, -5),
                     (-15, -15), (-5, -15), (5, -15)]

        for idx, (x, y) in enumerate(positions):
            ax.text(x, y, board[idx].value, fontsize=170)

        plt.pause(0.0001)

    def update_agent_gui(self, weights, msg):
        ax = self.perf_ax

        ax.cla()
        ax.set_title(msg)

        ax.plot(0, 0)
