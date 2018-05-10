"""
Reinforcement Learning based Tic-Tac-Toe
"""
import argparse

from rl_ttt import agents
from rl_ttt import environment


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--agent',
                        help='Type of agent to run',
                        choices=['q-learning', 'supervised'],
                        default='q-learning')

    parser.add_argument('--steps',
                        help='Number of training steps',
                        default=100)

    args = parser.parse_args()
    return args


def get_agent_cls(agent_type):
    available_agents = {
        'q-learning': agents.QLearningAgent,
        'supervised': None  # TODO: implement it!
    }

    return available_agents[agent_type]


def main():
    args = get_args()

    env = environment.TicTacToeEnv()

    agent_cls = get_agent_cls(args.agent)
    agent = agent_cls()

    agent.fit(env, nb_steps=10, verbose=2, visualize=True)


if __name__ == '__main__':
    main()
