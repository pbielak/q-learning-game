"""
Matplotlib-based GUI
"""
import matplotlib.pyplot as plt
from matplotlib import gridspec

from rl_ttt.gui import base
from rl_ttt import game


class WindowGUI(base.GUI):

    def __init__(self, cfg, stats):
        super(WindowGUI, self).__init__(cfg, stats)

        self.board = None
        self.agents = None
        self.game_outcomes = None

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

        self.board = {
            'ax': ax,
            'texts': texts
        }

    def _init_game_stats_gui(self, gs):
        ax = plt.subplot(gs[1])

        ax.plot(0, 0)
        brs = ax.bar(['X Wins', 'O Wins', 'Draws'], [0, 0, 0])
        ax.set_title('Episode / Game: 0')

        self.game_outcomes = {
            'ax': ax,
            game.GameStatus.X_WIN: brs[0],
            game.GameStatus.O_WIN: brs[1],
            game.GameStatus.DRAW: brs[2],
        }

    def _init_agent_stats_gui(self, gs):
        self.agents = {
            'r_ax': plt.subplot(gs[1:, 0]),
            'rd': None,

            'X': {
                'q_ax': plt.subplot(gs[3]),
                'mqd': None,
            },
            'O': {
                'q_ax': plt.subplot(gs[5]),
                'mqd': None,
            },
        }

        r_ax = self.agents['r_ax']
        rd = r_ax.plot(0, 0, label='Rewards')
        self.agents['rd'] = rd[0]

        for mt in ('X', 'O'):
            agent_stats = self.agents[mt]

            q_ax = agent_stats['q_ax']
            mqd = q_ax.plot(0, 0, color='r', linestyle='--', label='Mean-Q')
            agent_stats['mqd'] = mqd[0]

        plt.tight_layout()

    def update_env_gui(self, board, msg):
        self.board['ax'].set_title(msg)
        for idx, txt in enumerate(self.board['texts']):
            txt.set_text(board[idx].value)

        plt.pause(0.01)

    def _refresh_agent_gui(self):
        # Rewards
        r_ax = self.agents['r_ax']
        rewards = self.stats.agents['X']['rewards']
        x = list(range(len(rewards)))

        self.agents['rd'].set_data(x, rewards)

        msg = 'Total reward: {}'.format(rewards[-1])
        r_ax.set_title(msg)

        r_ax.relim()
        r_ax.autoscale_view(scalex=True, scaley=True)

        for mt in ('X', 'O'):
            agent_stats = self.agents[mt]

            # Q-values
            q_ax = agent_stats['q_ax']
            mean_qs = self.stats.agents[mt]['mean_qs']

            if mean_qs:
                agent_stats['mqd'].set_data(x, mean_qs)

                msg = '[{}] Mean Q: {}'.format(
                    mt, mean_qs[-1]
                )
                q_ax.set_title(msg)

            q_ax.relim()
            q_ax.autoscale_view(scalex=True, scaley=True)

    def _refresh_game_outcomes(self):
        ax = self.game_outcomes['ax']

        ax.set_title('Episode / Game: %d' % self.stats.episode)

        for gr_type, gr_value in self.stats.game_outcomes.items():
            self.game_outcomes[gr_type].set_height(gr_value)

        ax.relim()
        ax.autoscale_view(scaley=True)
