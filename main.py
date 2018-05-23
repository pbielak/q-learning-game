"""
Reinforcement Learning based Tic-Tac-Toe
"""
import matplotlib.pyplot as plt

from rl_ttt import agents as ag
from rl_ttt import environment
from rl_ttt import gui as vis
from rl_ttt import runner as run


def run_experiment(visualize):
    nb_steps = None  # 20000
    nb_episodes = 10000

    if visualize:
        gui = vis.TicTacToeGUI()
    else:
        gui = vis.GUI()

    env = environment.TicTacToeEnv(gui_callback=gui.update_env_gui)
    agents = [
        ag.QLearningAgent(name='AGENT_X', gui_callback=gui.update_agent_X_gui),
        ag.RandomAgent(name='AGENT_O', gui_callback=gui.update_agent_Y_gui)
    ]

    runner = run.Runner(agents, env)
    runner.train(nb_steps=nb_steps, nb_episodes=nb_episodes)


if __name__ == '__main__':
    run_experiment(visualize=False)
    plt.show(block=True)
