"""
Matplotlib-based GUI
"""
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np

from rl_ttt.gui import base
from rl_ttt import game


class TicTacToeGUI(base.GUI):

    def __init__(self, cfg):
        super(TicTacToeGUI, self).__init__(cfg)

        self.board_ax = None
        self.texts = None
        self.agent_stats = None
        self.game_stats = None

    def draw(self):
        plt.ion()
        fig = plt.figure(figsize=(14, 8))
        gs = gridspec.GridSpec(nrows=3, ncols=2, figure=fig,
                               width_ratios=[1, 1], height_ratios=[2, 1, 1])
        self._init_board_gui(gs)
        self._init_game_stats_gui(gs)
        self._init_agent_stats_gui(gs)
        plt.show()

    def _init_board_gui(self, gs):
        ax = plt.subplot(gs[0])

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

        texts = []
        for idx, (x, y) in enumerate(positions):
            texts.append(ax.text(x, y, '', fontsize=100))

        self.board_ax = ax
        self.texts = texts

    def _init_game_stats_gui(self, gs):
        ax = plt.subplot(gs[1])

        ax.plot(0, 0)
        brs = ax.bar(['X Wins', 'O Wins', 'Draws'], [0, 0, 0])
        ax.set_title('Episode / Game: 0')

        self.game_stats = {
            'ax': ax,
            game.GameStatus.X_WIN: brs[0],
            game.GameStatus.O_WIN: brs[1],
            game.GameStatus.DRAW: brs[2],
        }

    def _init_agent_stats_gui(self, gs):
        self.agent_stats = {
            'X': {
                'rewards': [],
                'mean-q': [],
                'ax': plt.subplot(gs[2]),
                'q_ax': plt.subplot(gs[3]),
            },
            'O': {
                'rewards': [],
                'mean-q': [],
                'ax': plt.subplot(gs[4]),
                'q_ax': plt.subplot(gs[5]),
            },
        }

        for mt in ('X', 'O'):
            agent_stats = self.agent_stats[mt]
            ax, q_ax = agent_stats['ax'], agent_stats['q_ax']

            rd = ax.plot(0, 0, label='Rewards')
            agent_stats['rd'] = rd[0]

            mqd = q_ax.plot(0, 0, color='r', linestyle='--', label='Mean-Q')
            agent_stats['mqd'] = mqd[0]

    def update_env_gui(self, board, msg):
        self.board_ax.set_title(msg)
        for idx, txt in enumerate(self.texts):
            txt.set_text(board[idx].value)

        plt.pause(0.0001)

    def update_agent_gui(self, reward, q_values, agent_name):
        agent_stats = self.agent_stats[agent_name]

        # Rewards
        rewards = agent_stats['rewards']
        if not rewards:
            rewards.append(reward)
        else:
            rewards.append(rewards[-1] + reward)

        x = list(range(len(rewards)))
        ax = agent_stats['ax']

        agent_stats['rd'].set_data(x, rewards)

        msg = '[{}] Total reward: {}'.format(
            agent_name, rewards[-1]
        )
        ax.set_title(msg)

        # Q-values
        q_ax = agent_stats['q_ax']
        mean_qs = agent_stats['mean-q']
        mean_qs.append(np.mean(q_values) if q_values else 'N/A')

        if q_values:
            agent_stats['mqd'].set_data(x, mean_qs)

        msg = '[{}] Mean Q: {}'.format(
            agent_name, mean_qs[-1]
        )
        q_ax.set_title(msg)

        # Refresh plots
        for _ax in (ax, q_ax):
            _ax.relim()
            _ax.autoscale_view(scalex=True, scaley=True)

        plt.tight_layout()

    def update_stats(self, stats):
        ax = self.game_stats['ax']

        ax.set_title('Episode / Game: %d' % stats.episode)

        for gr_type, gr_value in stats.game_results.items():
            self.game_stats[gr_type].set_height(gr_value)

        ax.relim()
        ax.autoscale_view(scaley=True)
