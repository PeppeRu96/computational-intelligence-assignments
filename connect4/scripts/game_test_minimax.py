from game_utility import *
from minimax import minimax_depth_limited, minimax_depth_limited_alpha_beta_pruning
from evaluation import *

# Wrappers for convenience
def human_vs_minimax(max_depth, estimate_f, estimate_name):
    human_player = build_human_player()

    minimax_play = lambda board, turn: minimax_depth_limited(board=board,
                                                             player=turn,
                                                             max_depth=max_depth,
                                                             eval_f=estimate_f
                                                             )

    minimax_player = Player(minimax_play, f"Minimax (Max depth: {max_depth}, {estimate_name})")

    play_connect4(human_player, minimax_player)

def human_vs_minimax_pruned(max_depth, estimate_f, estimate_name, move_sort_f, move_sort_name):
    human_player = build_human_player()

    minimax_play = lambda board, turn: minimax_depth_limited_alpha_beta_pruning(board=board,
                                                                                player=turn,
                                                                                max_depth=max_depth,
                                                                                eval_f=estimate_f,
                                                                                eval_for_sorting_moves_f=move_sort_f)

    minimax_player = Player(minimax_play,
                            f"Minimax with Alpha-Beta Pruning (Max depth: {max_depth}, Estimate: {estimate_name}, Function for sorting moves: {move_sort_name})")

    play_connect4(human_player, minimax_player)

def minimax_vs_minimax(max_depth1, pruning1, estimate_f1, estimate_name1, move_sort_f1, move_sort_name1,
                       max_depth2, pruning2, estimate_f2, estimate_name2, move_sort_f2, move_sort_name2):
    def build_minimax(max_depth, pruning, estimate_f, estimate_name, move_sort_f, move_sort_name):
        if pruning:
            minimax_play = lambda board, turn: minimax_depth_limited_alpha_beta_pruning(board=board,
                                                                                         player=turn,
                                                                                         max_depth=max_depth,
                                                                                         eval_f=estimate_f,
                                                                                         eval_for_sorting_moves_f=move_sort_f)
            name = f"Minimax with Alpha-Beta Pruning (Max depth: {max_depth}, Estimate: {estimate_name}, Function for sorting moves: {move_sort_name})"
        else:
            minimax_play = lambda board, turn: minimax_depth_limited(board=board,
                                                                      player=turn,
                                                                      max_depth=max_depth,
                                                                      eval_f=estimate_f
                                                                      )
            name = f"Minimax (Max depth: {max_depth1}, {estimate_name})"

        minimax_player = Player(minimax_play, name)
        return minimax_player

    minimax1 = build_minimax(max_depth1, pruning1, estimate_f1, estimate_name1, move_sort_f1, move_sort_name1)
    minimax2 = build_minimax(max_depth2, pruning2, estimate_f2, estimate_name2, move_sort_f2, move_sort_name2)

    play_connect4(minimax1, minimax2)


# Minimax (not pruned) experiments
def test1_human_vs_minimax_random_sampling_estim(max_depth, samples):
    estimate_f = lambda board, turn: random_rollout_estimate(board, turn, samples)
    estimate_name = f"Random rollout estimate with samples: {samples}"

    human_vs_minimax(max_depth, estimate_f, estimate_name)

def test2_human_vs_minimax_static_heuristic_estim(max_depth):
    estimate_f = lambda board, turn: heuristic_static_magic_square_estimate(board)
    estimate_name = f"Static heuristic with magic square estimate"

    human_vs_minimax(max_depth, estimate_f, estimate_name)

# Minimax with Alpha-Beta pruning experiments
def test3_human_vs_minimax_pruned(max_depth, samples1, samples2):
    """
        minimax alpha-beta pruned with a random sampling estimation of non-terminal nodes (when reached max_depth)
        plus random sampling estimation for sorting the moves
        :param max_depth:
        :param samples:
        :return:
        """
    estimate_f = lambda board, turn: random_rollout_estimate(board, turn, samples1)
    estimate_name = f"Random rollout estimate with samples: {samples1}"

    move_sort_f = lambda board, turn: random_rollout_estimate(board, turn, samples2)
    move_sort_name = f"Random rollout estimate with samples: {samples2}"

    human_vs_minimax_pruned(max_depth, estimate_f, estimate_name, move_sort_f, move_sort_name)

def test4_human_vs_minimax_pruned(max_depth, samples):
    """
    minimax alpha-beta pruned with a random sampling estimation of non-terminal nodes (when reached max_depth)
    plus static heuristic magic square estimation for sorting the moves
    :param max_depth:
    :param samples:
    :return:
    """
    estimate_f = lambda board, turn: random_rollout_estimate(board, turn, samples)
    estimate_name = f"Random rollout estimate with samples: {samples}"

    move_sort_f = lambda board, turn: heuristic_static_magic_square_estimate(board)
    move_sort_name = f"Static heuristic magic square estimation"

    human_vs_minimax_pruned(max_depth, estimate_f, estimate_name, move_sort_f, move_sort_name)

def test5_human_vs_minimax_pruned(max_depth):
    """
        minimax alpha-beta pruned with a static heuristic magic square estimation of non-terminal nodes (when reached max_depth)
        plus static heuristic magic square estimation for sorting the moves
        :param max_depth:
        :param samples:
        :return:
    """
    estimate_f = lambda board, turn: heuristic_static_magic_square_estimate(board)
    estimate_name = f"Static heuristic magic square estimation"

    move_sort_f = lambda board, turn: heuristic_static_magic_square_estimate(board)
    move_sort_name = f"Static heuristic magic square estimation"

    human_vs_minimax_pruned(max_depth, estimate_f, estimate_name, move_sort_f, move_sort_name)

def test6_minimax_vs_minimax_pruned(max_depth1, max_depth2):
    estimate_f = lambda board, turn: heuristic_static_magic_square_estimate(board)
    estimate_name = f"Static heuristic with magic square estimate"
    minimax_vs_minimax(max_depth1=max_depth1,
                       pruning1=False,
                       estimate_f1=estimate_f,
                       estimate_name1=estimate_name,
                       move_sort_f1=None,
                       move_sort_name1=None,
                       max_depth2=max_depth2,
                       pruning2=True,
                       estimate_f2=estimate_f,
                       estimate_name2=estimate_name,
                       move_sort_f2=estimate_f,
                       move_sort_name2=estimate_name
                       )

if __name__ == "__main__":
    # Minimax (not pruned)
    #test1_human_vs_minimax_random_sampling_estim(max_depth=3, samples=5)
    #test2_human_vs_minimax_static_heuristic_estim(max_depth=4)

    # Minimax with Alpha-Beta pruning
    #test3_human_vs_minimax_pruned(max_depth=3, samples1=5, samples2=3)
    #test4_human_vs_minimax_pruned(max_depth=3, samples=10)
    #test5_human_vs_minimax_pruned(max_depth=6)
    test6_minimax_vs_minimax_pruned(max_depth1=4, max_depth2=6)
