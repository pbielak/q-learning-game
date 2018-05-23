"""
Tic-Tac-Toe GUI
"""
import matplotlib.pyplot as plt
import seaborn as sns


class GUI(object):
    def update_env_gui(self, **kwargs):
        pass

    def update_agent_X_gui(self, weights, reward, msg):
        pass

    def update_agent_Y_gui(self, weights, reward, msg):
        pass


class TicTacToeGUI(GUI):

    def __init__(self):
        self.board_ax = None

        self.perf_X_ax = None
        self.weights_X_ax = None
        self.perf_Y_ax = None
        self.weights_Y_ax = None

        self.agent_rewards = {}
        self._init_gui()

    def _init_gui(self):
        plt.ion()
        fig = plt.figure(figsize=(14, 7))
        self.board_ax = plt.subplot2grid((2, 3), (0, 0), rowspan=2, fig=fig)

        self.perf_X_ax = plt.subplot2grid((2, 3), (0, 1), fig=fig)
        self.weights_X_ax = plt.subplot2grid((2, 3), (0, 2), fig=fig)

        self.perf_Y_ax = plt.subplot2grid((2, 3), (1, 1), fig=fig)
        self.weights_Y_ax = plt.subplot2grid((2, 3), (1, 2), fig=fig)

        self.agent_rewards['X'] = {'step': [0], 'cummulative': [0]}
        self.agent_rewards['Y'] = {'step': [0], 'cummulative': [0]}
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

    def update_agent_X_gui(self, weights, reward, msg):
        self.agent_rewards['X']['step'].append(reward)
        self.agent_rewards['X']['cummulative'].append(
            self.agent_rewards['X']['cummulative'][-1] + reward
        )
        self._update_agent_gui(weights, self.agent_rewards['X'],
                               msg, perf_ax=self.perf_X_ax,
                               weights_ax=self.weights_X_ax)

    def update_agent_Y_gui(self, weights, reward, msg):
        self.agent_rewards['Y']['step'].append(reward)
        self.agent_rewards['Y']['cummulative'].append(
            self.agent_rewards['Y']['cummulative'][-1] + reward
        )
        self._update_agent_gui(weights, self.agent_rewards['Y'],
                               msg, perf_ax=self.perf_Y_ax,
                               weights_ax=self.weights_Y_ax)

    @staticmethod
    def _update_agent_gui(weights, rewards, msg, perf_ax, weights_ax):
        perf_ax.cla()
        perf_ax.set_title(msg)
        perf_ax.plot(list(range(0, len(rewards['step']))), rewards['step'],
                     label='Current reward', linestyle='--')

        perf_ax.plot(list(range(0, len(rewards['cummulative']))),
                     rewards['cummulative'], label='Cummulative reward')

        perf_ax.legend()

        if weights:
            weights_ax.cla()
            sns.heatmap(weights, ax=weights_ax, cbar=False,
                        xticklabels=False, yticklabels=False)
