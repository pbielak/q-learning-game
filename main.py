"""
Reinforcement Learning based Tic-Tac-Toe
"""
import matplotlib.pyplot as plt
from rl_ttt import agents
from rl_ttt import environment
from rl_ttt import gui


def main():
    nb_steps = 20

    g = gui.TicTacToeGUI()
    env = environment.TicTacToeEnv(gui_callback=g.update_env_gui)
    agent = agents.QLearningAgent(gui_callback=g.update_agent_gui)

    agent.fit(env, nb_steps=nb_steps, verbose=0, visualize=True)


if __name__ == '__main__':
    main()
    plt.show(block=True)
