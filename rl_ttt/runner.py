"""
Train/test runner for RL
"""


class Runner(object):

    def __init__(self, agents, environment, stats, gui_callback):
        self.agents = agents
        self.env = environment
        self.stats = stats
        self.gui_callback = gui_callback

    def train(self, nb_episodes):
        observation = None
        reward = None
        done = None
        info = None

        while self.stats.episode < nb_episodes:
            if observation is None:
                observation = self.env.reset()
                self.env.render()
                self.stats.episode += 1

            for agent in self.agents:
                action = agent.forward(observation)
                observation, reward, done, info = self.env.step(action)
                self.env.render()
                if done:
                    break
                assert reward == 0
                agent.backward(reward, terminal=False)

            if done:
                self.stats.increment_game_result(info['status'])

                for agent, reward_multiplier in zip(self.agents, [1, -1]):
                    agent.forward(observation)
                    agent.backward(reward_multiplier * reward, terminal=True)
                observation = None

            self.gui_callback()

    def test(self):
        raise NotImplementedError()

