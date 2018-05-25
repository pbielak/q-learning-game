"""
Module for configuration related classes / functions etc.
"""
from collections import namedtuple


class ExperimentConfig(object):

    def __init__(self, nb_episodes, visualize):
        self.nb_episodes = nb_episodes
        self.visualize = visualize
        self.agents_configs = []

    def add_q_learning_agent(self, marker_type, learning_rate, discount_factor,
                             eps, load_weights, save_weights):
        self.agents_configs.append(
            QLearningAgentConfig(marker_type, learning_rate, discount_factor,
                                 eps, load_weights, save_weights)
        )

        assert len(self.agents_configs) <= 2
        return self

    def add_random_agent(self, marker_type):
        self.agents_configs.append(RandomAgentConfig(marker_type))
        assert len(self.agents_configs) <= 2
        return self


QLearningAgentConfig = namedtuple('QLearningAgentConfig', ['marker_type',
                                                           'learning_rate',
                                                           'discount_factor',
                                                           'eps',
                                                           'load_weights',
                                                           'save_weights'])

RandomAgentConfig = namedtuple('RandomAgentConfig', ['marker_type'])
