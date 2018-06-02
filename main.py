"""
Reinforcement Learning based Tic-Tac-Toe
"""
import matplotlib.pyplot as plt

from rl_ttt import agents as ttt_agents
from rl_ttt import config as ttt_cfg
from rl_ttt import environment as ttt_env
from rl_ttt import game as ttt_game
from rl_ttt import runner as ttt_runner
from rl_ttt import stats as ttt_stats
from rl_ttt import gui as ttt_gui


def get_cfg():
    cfg = ttt_cfg.ExperimentConfig(nb_episodes=1000,
                                   visualize=False)

    cfg.add_q_learning_agent(marker_type='X',
                             learning_rate=0.001,
                             discount_factor=0.6,
                             eps=0.2,
                             load_weights=False,
                             save_weights=True,
                             batch_mode=True)

    cfg.add_random_agent(marker_type='O')

    return cfg


def make_agents(cfg, stats):
    agents = []

    for agent_cfg in cfg.agents_configs:
        if isinstance(agent_cfg, ttt_cfg.QLearningAgentConfig):
            agent = ttt_agents.q_learning.from_config(agent_cfg, stats)

            if agent_cfg.load_weights:
                agent.load_q_values(
                    'data/%s_q_values.save' % agent_cfg.marker_type
                )
                print('Loaded q values for %s!' % agent_cfg.marker_type)

            agents.append(agent)
        elif isinstance(agent_cfg, ttt_cfg.RandomAgentConfig):
            agent = ttt_agents.random.from_config(agent_cfg, stats)
            agents.append(agent)
        else:
            raise RuntimeError('Config %s not recognized' % type(agent_cfg))

    return agents


def get_gui(cfg, stats):
    if cfg.visualize:
        return ttt_gui.window.WindowGUI(cfg, stats)

    return ttt_gui.console.ConsoleGUI(cfg, stats)


def on_experiment_end(agents, cfg):
    agents_map = {}
    for agent in agents:
        agents_map[agent.marker_type] = agent

    for agent_cfg in cfg.agents_configs:
        if isinstance(agent_cfg, ttt_cfg.QLearningAgentConfig) and \
                agent_cfg.save_weights:
            agents_map[agent_cfg.marker_type].save_q_values(
                'data/%s_q_values.save' % agent_cfg.marker_type
            )
            print('Saved q values for %s!' % agent_cfg.marker_type)


def run_experiment():
    cfg = get_cfg()

    stats = ttt_stats.Stats()

    game = ttt_game.TicTacToe()
    gui = get_gui(cfg, stats)
    env = ttt_env.TicTacToeEnv(game=game, gui_callback=gui.update_env_gui)

    agents = make_agents(cfg, stats)

    runner = ttt_runner.Runner(agents, env, stats, gui.refresh)

    gui.draw()
    runner.train(nb_episodes=cfg.nb_episodes)

    on_experiment_end(agents, cfg)

    stats.summary()

    if cfg.visualize:
        plt.show(block=True)


if __name__ == '__main__':
    run_experiment()
