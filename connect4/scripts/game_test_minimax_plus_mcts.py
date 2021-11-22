import math

from game_utility import *
from minimax_plus_mcts import minimax_plus_montecarlo_tree_search
from montecarlo_tree_search import rollout_random_sampling, rollout_static_heuristic_magic_square
from evaluation import *

def human_vs_minimax_plus_mcts(max_depth, move_sort_f, move_sort_name, iterations, rollout_f, rollout_name, c_param):
    human_player = build_human_player()

    play = lambda board, turn: minimax_plus_montecarlo_tree_search(board=board,
                                                                   player=turn,
                                                                   max_depth=max_depth,
                                                                   sort_move_f=move_sort_f,
                                                                   iterations=iterations,
                                                                   rollout_f=rollout_f,
                                                                   c_param=c_param)

    player = Player(play, f"Minimax plus MCTS (Max depth: {max_depth}, Moves sorted by: {move_sort_name}, MCTS iterations: {iterations}, Rollout: {rollout_name})")

    play_connect4(human_player, player, show_taken_time=True)

def test1_human_vs_minimax_plus_mcts_random_rollout(max_depth, iterations, samples, c_param):
    move_sort_f = lambda board, turn: rollout_static_heuristic_magic_square(board)
    move_sort_name = f"Static heuristic with magic square estimate"

    rollout_f = lambda board, node: rollout_random_sampling(board, node, samples)
    rollout_name = f"Random sampling estimate (samples: {samples})"

    human_vs_minimax_plus_mcts(max_depth, move_sort_f, move_sort_name, iterations, rollout_f, rollout_name, c_param)

def test2_human_vs_minimax_plus_mcts_static_heuristic_rollout(max_depth, iterations, c_param):
    move_sort_f = lambda board, turn: heuristic_static_magic_square_estimate(board)
    move_sort_name = f"Static heuristic with magic square estimate"

    rollout_f = lambda board, node: heuristic_static_magic_square_estimate(board)
    rollout_name = f"Static heuristic with magic square estimate"

    human_vs_minimax_plus_mcts(max_depth, move_sort_f, move_sort_name, iterations, rollout_f, rollout_name, c_param)

if __name__ == "__main__":
    #test1_human_vs_minimax_plus_mcts_random_rollout(max_depth=2, iterations=30, samples=3, c_param=math.sqrt(2))
    test2_human_vs_minimax_plus_mcts_static_heuristic_rollout(max_depth=3, iterations=30, c_param=math.sqrt(2))
