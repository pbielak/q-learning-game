"""
Tic-Tac-Toe GUI
"""
import matplotlib.pyplot as plt


class GUI(object):
    def update_env_gui(self, **kwargs):
        pass

    def update_agent_gui(self, reward, agent_name):
        pass


class TicTacToeGUI(GUI):

    def __init__(self):
        self.board_ax = None
        self.agent_rewards = None
        self._init_gui()

    def _init_gui(self):
        plt.ion()
        fig = plt.figure(figsize=(14, 7))
        self.board_ax = plt.subplot2grid((2, 3), (0, 0), rowspan=2, fig=fig)
        self.agent_rewards = {
            'X': {
                'rewards': [0],
                'ax': plt.subplot2grid((2, 3), (0, 1), colspan=2, fig=fig)},
            'O': {
                'rewards': [0],
                'ax': plt.subplot2grid((2, 3), (1, 1), colspan=2, fig=fig)},
        }
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
            ax.text(x, y, board[idx].value, fontsize=100)

        plt.pause(0.0001)

    def update_agent_gui(self, reward, agent_name):
        rewards = self.agent_rewards[agent_name]['rewards']
        rewards.append(rewards[-1] + reward)

        ax = self.agent_rewards[agent_name]['ax']

        ax.cla()
        ax.plot(list(range(0, len(rewards))), rewards)

        msg = 'Cumulative reward: {}'.format(rewards[-1])
        ax.set_title(msg)
