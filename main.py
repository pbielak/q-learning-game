"""
Reinforcement Learning based Tic-Tac-Toe
"""
import matplotlib.pyplot as plt
from rl_ttt import agents
from rl_ttt import environment
from rl_ttt import gui
from rl_ttt import runner


def main():
    nb_steps = 100

    g = gui.TicTacToeGUI()
    env = environment.TicTacToeEnv(gui_callback=g.update_env_gui)
    agent = agents.QLearningAgent(name='AGENT_X', gui_callback=g.update_agent_X_gui)
    agent2 = agents.QLearningAgent(name='AGENT_O', gui_callback=g.update_agent_Y_gui)

    run = runner.Runner([agent, agent2], env)

    run.train(nb_steps)


if __name__ == '__main__':
    main()
    plt.show(block=True)
