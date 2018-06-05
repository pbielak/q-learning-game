"""
Module for random games aggregation
"""
import collections

from tqdm import tqdm


class Aggregator(object):

    def __init__(self, history=None, filename=None):
        self.aggregated = collections.defaultdict(lambda: dict(x_wins=0,
                                                               o_wins=0,
                                                               draws=0,
                                                               total=0))
        self.history = self._initialize_history(history, filename)
        self.pb = None

    def _initialize_history(self, history, filename):
        if not history and not filename:
            raise RuntimeError('At least history or filename must be filled')
        elif history and filename:
            raise RuntimeError('Cannot handle both history and filename filled')
        elif history:
            return history
        else:
            return self._load(filename)

    def _load(self, filename):
        with open(filename, 'r') as f:
            lines = f.read().split('\n')

        history = []

        for line in tqdm(lines, desc='Loading', position=0):
            reward, episode_data = line.split(';', maxsplit=1)
            reward = int(reward)

            episode_data_entries = [h[1:-1] for h in episode_data.split(';')]
            episode_data = []

            for he in episode_data_entries:
                a, s = he.split(',', maxsplit=1)
                a = int(a)
                s = tuple([sp[1:-1] for sp in
                           s.strip().replace(' ', '')[1:-1].split(',')])

                episode_data.append((s, a))

            history.append((reward, episode_data))

        return history

    def run(self):
        reward_mapping = {
            1: 'x_wins',
            0: 'draws',
            -1: 'o_wins'
        }

        for reward, episode_data in tqdm(self.history, desc='Aggregating',
                                         position=0):
            for ed in episode_data:
                state, action = ed
                self.aggregated[(state, action)][reward_mapping[reward]] += 1
                self.aggregated[(state, action)]['total'] += 1

    def _aggregation_coeff(self, x_wins, o_wins, draws,
                           x_weight=1, o_weight=-1, draw_weight=0.001):
        total = x_wins + o_wins + draws
        coeff = x_wins * x_weight + o_wins * o_weight + draws * draw_weight
        return coeff / total

    def summary(self, where=lambda v: True):
        print('\n\n')
        print('Summary:')
        for k, v in sorted(self.aggregated.items(), key=lambda i: i[1]['total']):
            if where(v):
                x_wins = v['x_wins']
                o_wins = v['o_wins']
                draws = v['draws']
                coeff = self._aggregation_coeff(x_wins, o_wins, draws)

                print(f'{k} => {v} (coeff={coeff})')

    def save_aggregation(self, filename, where=lambda v: True):
        lines = []

        aggregated = {k: v for k, v in self.aggregated.items() if where(v)}
        for k, v in tqdm(aggregated.items(), desc='Saving aggregation',
                         position=0):

            k = str(k).replace(' ', '')
            coeff = self._aggregation_coeff(v['x_wins'],
                                            v['o_wins'],
                                            v['draws'])
            lines.append(f'{k};{coeff}')

        with open(filename, 'w') as f:
            f.write('\n'.join(lines))

