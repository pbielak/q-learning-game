"""
Module for SL model
"""
from rl_ttt import supervised_learning as ttt_sl


def main():
    # filename = 'data/random_games.save'
    # rgg = ttt_sl.generator.RandomGameGenerator()
    # rgg.run(nb_episodes=200000)
    # rgg.save(filename, append=False)
    #
    # agg = ttt_sl.aggregator.Aggregator(filename=filename)
    # agg.run()
    #
    # where = lambda v: v['total'] > 10
    # agg.summary(where=where)
    # agg.save_aggregation('data/nn_data.save', where=where)

    tr = ttt_sl.trainer.Trainer()
    tr.load('data/nn_data.save')
    tr.train()


if __name__ == '__main__':
    main()
