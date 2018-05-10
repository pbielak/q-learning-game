"""
Agent implementations
"""
from rl import core


class QLearningAgent(core.Agent):
    compiled = True

    def __init__(self):
        super(QLearningAgent, self).__init__()

    def forward(self, observation):
        # TODO
        pass

    def backward(self, reward, terminal):
        # TODO
        pass

    def load_weights(self, filepath):
        return

    def save_weights(self, filepath, overwrite=False):
        return

    def compile(self, optimizer, metrics=[]):
        return

    @property
    def layers(self):
        return
