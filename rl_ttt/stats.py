"""
Module for statistics class
"""
import matplotlib.pyplot as plt
import numpy as np

from rl_ttt import game


class Stats(object):

    def __init__(self):
        self.episode = 0
        self.game_outcomes = {
            game.GameStatus.X_WIN: 0,
            game.GameStatus.O_WIN: 0,
            game.GameStatus.DRAW: 0,
        }
        self.agents = {
            game.FieldStates.X_MARKER.value: {
                'rewards': [(0, 0)],
                'mean_qs': []
            },
            game.FieldStates.O_MARKER.value: {
                'rewards': [(0, 0)],
                'mean_qs': []
            }
        }

    def increment_game_result(self, outcome_type):
        self.game_outcomes[outcome_type] += 1

    def add_reward(self, marker_type, reward):
        rewards = self.agents[marker_type]['rewards']
        r = rewards[-1][1] + reward

        rewards.append((self.episode, r))

    def add_mean_q(self, marker_type, q_values):
        if q_values:
            mean_qs = self.agents[marker_type]['mean_qs']
            mean_q = np.mean(q_values)
            mean_qs.append((self.episode, mean_q))

    def summary(self):
        print(self._summarize_game_outcomes())
        print(self._summarize_rewards())
        print(self._summarize_mean_qs())
        self._make_summary_plots()

    def _summarize_game_outcomes(self):
        x_wins = self.game_outcomes[game.GameStatus.X_WIN]
        o_wins = self.game_outcomes[game.GameStatus.O_WIN]
        draws = self.game_outcomes[game.GameStatus.DRAW]

        total_games = x_wins + o_wins + draws

        fmt_str = "\n" \
                  "X_WINS\t{}\t{} (%)\n" \
                  "O_WINS\t{}\t{} (%)\n" \
                  "DRAW\t{}\t{} (%)\n"

        msg = fmt_str.format(
            x_wins, np.round(100 * x_wins / total_games),
            o_wins, np.round(100 * o_wins / total_games),
            draws, np.round(100 * draws / total_games),
        )

        return msg

    def _summarize_rewards(self):
        msg = 'Total rewards:\n'
        for mt, ag_stats in self.agents.items():
            _, r = ag_stats['rewards'][-1]
            msg += f'{mt} => {r}\n'

        return msg

    def _summarize_mean_qs(self):
        msg = 'Mean-Qs:\n'
        for mt, ag_stats in self.agents.items():
            mean_qs = ag_stats['mean_qs']

            if mean_qs:
                _, start_mq = mean_qs[0]
                _, end_mq = mean_qs[-1]
                growth = np.round((end_mq - start_mq) / start_mq * 100.0, 2)

                msg += f'{mt} => {start_mq}, {end_mq} ({growth} (%))\n'
            else:
                msg += f'{mt} => N/A\n'

        return msg

    def _make_summary_plots(self):
        fig = plt.figure(figsize=(10, 7))

        ax1 = fig.add_subplot(2, 2, 1)
        rewards = np.array(self.agents['X']['rewards'])
        x, y = rewards[:, 0], rewards[:, 1]
        ax1.plot(x, y, color='b', label='Rewards')
        ax1.set_title(f'[X] Total reward: {rewards[-1][1]}')

        ax2 = fig.add_subplot(2, 2, 2)
        wins = [self.game_outcomes[game.GameStatus.X_WIN],
                self.game_outcomes[game.GameStatus.O_WIN],
                self.game_outcomes[game.GameStatus.DRAW]]
        ax2.bar(['X Wins', 'O Wins', 'Draws'], wins)
        for idx, w in enumerate(wins):
            ax2.text(idx, w + 0.5, f'{w} ({np.round(100 * w/sum(wins))} %)',
                     ha='center')
        ax2.set_title('Game outcomes')

        for idx, mt in enumerate(('X', 'O')):
            ax = fig.add_subplot(2, 2, 3 + idx)
            mean_qs = self.agents[mt]['mean_qs']

            if mean_qs:
                mean_qs = np.array(mean_qs)
                x, y = mean_qs[:, 0], mean_qs[:, 1]
                title = f'[{mt}] Mean-Q: {mean_qs[-1][1]}'
            else:
                x, y = 0, 0
                title = f'[{mt}] Mean-Q: N/A'

            ax.plot(x, y, color='r', linestyle='--', label='Mean-Q')
            ax.set_title(title)

        plt.show()
