"""
Train/test runner for RL
"""


class Runner(object):

    def __init__(self, agents, environment):
        self.agents = agents
        self.env = environment
        self.won = {'X': 0, 'O': 0}

    def train(self, nb_steps):
        step = 0
        observation = None
        reward = None
        done = None

        while step < nb_steps:
            print('ENV: Step = ', step)
            if observation is None:
                observation = self.env.reset()
                self.env.render()

            for agent in self.agents:
                action = agent.forward(observation)
                observation, reward, done, info = self.env.step(action)
                self.env.render()
                if done:
                    break
                agent.backward(reward, terminal=False)

            if done:
                if reward == 1: # X won
                    self.won['X'] += 1
                else:
                    self.won['O'] += 1

                for agent, reward_multiplier in zip(self.agents, [1, -1]):
                    agent.forward(observation)
                    agent.backward(reward_multiplier * reward, terminal=True)
                observation = None

            step += 1

        print('Wins:', self.won)

    def test(self):
        raise NotImplementedError()
