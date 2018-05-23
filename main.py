"""
Reinforcement Learning based Tic-Tac-Toe
"""
import matplotlib.pyplot as plt

from rl_ttt import agents as ttt_agents
from rl_ttt import environment as ttt_env
from rl_ttt import game as ttt_game
from rl_ttt import gui as ttt_gui
from rl_ttt import runner as ttt_runner


def run_experiment(visualize=False, load_weights=False, save_weights=False):
    nb_steps = None  # 20000
    nb_episodes = 1000

    if visualize:
        gui = ttt_gui.TicTacToeGUI()
    else:
        gui = ttt_gui.GUI()

    game = ttt_game.TicTacToe()
    env = ttt_env.TicTacToeEnv(game=game, gui_callback=gui.update_env_gui)

    q_learning_agent = ttt_agents.QLearningAgent(
        name='AGENT_X',
        gui_callback=lambda r: gui.update_agent_gui(r, 'X')
    )

    if load_weights:
        q_learning_agent.load_q_values('data/q_values.save')
        print('Loaded q values!')

    random_agent = ttt_agents.RandomAgent(
        name='AGENT_O',
        gui_callback=lambda r: gui.update_agent_gui(r, 'O')
    )

    agents = [q_learning_agent, random_agent]

    runner = ttt_runner.Runner(agents, env)
    runner.train(nb_steps=nb_steps, nb_episodes=nb_episodes)

    if save_weights:
        q_learning_agent.save_q_values('data/q_values.save')
        print('Saved q values!')


if __name__ == '__main__':
    run_experiment(visualize=False,
                   load_weights=False,
                   save_weights=True)
    plt.show(block=True)
