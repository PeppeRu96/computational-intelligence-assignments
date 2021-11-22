from game_utility import *
from montecarlo_tree_search import *

def human_vs_mcts(rollout_f, rollout_name, iterations = None, max_time = None, c_param=0.1, show_tree=False, show_only_sub_tree=False):
    human_player = build_human_player()

    end_str = f"Iterations: {iterations}" if iterations is not None else f"Max time: {max_time}"
    montecarlo_name = f"Montecarlo Tree Search ({end_str}, Rollout: {rollout_name})"

    mcts_wrapper = MCTS(player_n=-1,
                        iterations=iterations,
                        max_time=max_time,
                        rollout_f=rollout_f,
                        draw_tree=show_tree,
                        draw_only_sub_tree=show_only_sub_tree,
                        c_param=c_param)

    mcts_player = Player(mcts_wrapper.play, montecarlo_name, mcts_wrapper.set_opponent_move)

    play_connect4(human_player, mcts_player, show_taken_time=True)

def test1_human_vs_mcts_random_sampling(samples, iterations=None, max_time=None, c_param=0.1, show_tree=False, show_only_sub_tree=False):
    rollout_f = lambda board, node: rollout_random_sampling(board, node, samples)
    rollout_name = f"Random sampling estimate (samples: {samples})"

    human_vs_mcts(rollout_f, rollout_name, iterations, max_time, c_param, show_tree, show_only_sub_tree)

def test2_human_vs_mcts_static_heuristic(iterations=None, max_time=None, c_param=0.1, show_tree=False, show_only_sub_tree=False):
    rollout_f = lambda board, node: rollout_static_heuristic_magic_square(board)
    rollout_name = f"Static heuristic magic square estimation)"

    human_vs_mcts(rollout_f, rollout_name, iterations, max_time, c_param, show_tree, show_only_sub_tree)

if __name__ == "__main__":
    #test1_human_vs_mcts_random_sampling(samples=3, iterations=250, c_param=math.sqrt(2), show_tree=False, show_only_sub_tree=True)
    test2_human_vs_mcts_static_heuristic(iterations=1000, c_param=math.sqrt(2), show_tree=False, show_only_sub_tree=True)
