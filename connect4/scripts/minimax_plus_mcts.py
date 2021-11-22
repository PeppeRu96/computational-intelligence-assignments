from montecarlo_tree_search import _montecarlo_tree_search
from minimax import *

def minimax_plus_montecarlo_tree_search(board, player, max_depth, sort_move_f, iterations, rollout_f, c_param):
    def mcts_eval(b, p):
        return _montecarlo_tree_search(
            board=b,
            player=p,
            G=None,
            start_node=None,
            iterations=iterations,
            max_time=None,
            rollout_f=rollout_f,
            c_param=c_param
        ).reward

    return minimax_depth_limited_alpha_beta_pruning(board, player, max_depth, mcts_eval, sort_move_f)