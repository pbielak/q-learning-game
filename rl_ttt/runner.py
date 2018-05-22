"""
Train/test runner for RL
"""
from rl_ttt import utils


class Runner(object):

    def __init__(self, agents, environment):
        self.agents = agents
        self.env = environment
        self.won = {'X': 0, 'O': 0}

    def train(self, nb_steps=None, nb_episodes=None):
        sc = utils.get_stop_condition(nb_steps, nb_episodes)

        step = 0
        episode = 0

        observation = None
        reward = None
        done = None

        while sc(step, episode):
            print('[Runner] Step =', step)
            if observation is None:
                observation = self.env.reset()
                self.env.render()
                episode += 1
                print('[Runner] Episode =', episode)

            for agent in self.agents:
                action = agent.forward(observation)
                observation, reward, done, info = self.env.step(action)
                self.env.render()
                if done:
                    break
                assert reward == 0
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
