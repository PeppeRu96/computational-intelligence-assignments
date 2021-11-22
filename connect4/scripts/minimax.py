from evaluation import *
import math

# Standard Minimax depth-limited
def _minimax_depth_limited(board, player, max_depth, eval_f):
    """
    Minimax Depth-Limited standard
    :param board:
    :param player:
    :param max_depth:
    :param eval_f: A function which takes in two parameters (board, player) and returns a score (positive for player 1, negative for player -1)
    :return: A tuple (best_column, best_score)
    """

    # If we've reached max_depth, we estimate the current node with the provided eval_f function
    if max_depth <= 0:
        val = eval_f(board, player)
        return None, val

    # Check if it is a leaf node (won or lost)
    val = check_winner(board)
    if val != 0:
        val = val * (max_depth + 1)
        return None, val
        # Amplify the score with depth (it will prefer less depth good moves and it will penalize a lot less depth losses)
        # Without this, imagine what it would happen if this move were strictly to be done (otherwise the opponent would win)
        # With this, imagine what it would happen if this move were strictly to be done, the other moves will be highly penalized
        # Even when the forced move gives back a bad score, that score will be at least better than all the other moves and it will do it.

    possible_moves = valid_moves(board)
    if len(possible_moves) == 0:
        return None, val

    # Expand all possible moves
    move_evals = list()
    for move in possible_moves:
        play(board, move, player)
        _, val = _minimax_depth_limited(board, player * -1, max_depth - 1, eval_f)
        take_back(board, move)

        move_evals.append((move, val))

    # We have to reason:
    # eval_f and check_winner will return positive values if player 1 wins, negative values if player -1 wins
    # This means that if we are player1 we need to maximize, else if we are player -1, we need to minimize
    if player == 1:
        best_move = max(move_evals, key=lambda k: k[1])
    else:
        best_move = min(move_evals, key=lambda k: k[1])

    return best_move

# Just a wrapper for returning only the move to be done (without the associated score)
def minimax_depth_limited(board, player, max_depth, eval_f):
    """
    Minimax Depth-Limited standard
    :param board:
    :param player:
    :param max_depth:
    :param eval_f: A function which takes in two parameters (board, player) and returns a score (positive for player 1, negative for player -1)
    :return: The move
    """
    obl_m = obliged_move(board, player)
    if obl_m is not None:
        return obl_m
    move = _minimax_depth_limited(board, player, max_depth, eval_f)[0]
    return move


# Minimax depth-limited with Alpha-Beta pruning
def _minimax_depth_limited_alpha_beta_pruning(board, player, max_depth, eval_f, eval_for_sorting_moves_f, alpha, beta):
    """
    Minimax depth-limited with Alpha-Beta pruning
    :param board:
    :param player:
    :param max_depth:
    :param eval_f: A function (board, player) -> score (positive for player 1, negative for player -1)
    :param eval_for_sorting_moves_f: A function needed to sort the moves to try (best first)
    in order for the alpha-beta-pruning to be effective
    :param alpha:
    :param beta:
    :return: A tuple (best_column, best_score)
    """

    # If we've reached max_depth, we estimate the current node with the provided eval_f function
    if max_depth <= 0:
        val = eval_f(board, player)
        return None, val

    # Check if it is a leaf node (won or lost)
    val = check_winner(board)
    if val != 0:
        val = val * (max_depth + 1)
        return None, val

    possible_moves = valid_moves(board)
    if len(possible_moves) == 0:
        return None, val

    # Evaulating the moves against some evaluation function to sort them (best first)
    move_evals = list()
    for move in possible_moves:
        play(board, move, player)
        eval = eval_for_sorting_moves_f(board, -player)
        take_back(board, move)
        move_evals.append((move, eval))

    if player == 1:  # maximizing player
        # Sort the moves to analyze the best first..
        move_evals.sort(reverse=True, key=lambda k: k[1])

        # Expand moves
        best_val = -math.inf
        best_move = None
        for move, _ in move_evals:
            play(board, move, player)
            _, val = _minimax_depth_limited_alpha_beta_pruning(board, player * -1, max_depth - 1, eval_f,
                                                               eval_for_sorting_moves_f, alpha, beta)
            take_back(board, move)

            if val > best_val:
                best_val = val
                best_move = move

            # alpha = max(alpha, bestVal)
            alpha = max(alpha, val)

            if beta <= alpha:
                break

        return (best_move, best_val)
    else:  # minimizing player
        # Sort the moves to analyze the best first..
        move_evals.sort(reverse=False, key=lambda k: k[1])

        moves = list()

        # Expand moves
        best_val = math.inf
        best_move = None
        for move, _ in move_evals:
            play(board, move, player)
            _, val = _minimax_depth_limited_alpha_beta_pruning(board, player * -1, max_depth - 1, eval_f,
                                                               eval_for_sorting_moves_f, alpha, beta)
            take_back(board, move)

            if val < best_val:
                best_val = val
                best_move = move

            # beta = min(beta, bestVal)
            beta = min(beta, val)

            if beta <= alpha:
                break

        return (best_move, best_val)

# Just a wrapper for returning only the move to be done (without the associated score)
def minimax_depth_limited_alpha_beta_pruning(board, player, max_depth, eval_f, eval_for_sorting_moves_f):
    """
    Minimax depth-limited with Alpha-Beta pruning
    :param board:
    :param player:
    :param max_depth:
    :param eval_f: A function (board, player) -> score (positive for player 1, negative for player -1)
    :param eval_for_sorting_moves_f: A function needed to sort the moves to try (best first)
    in order for the alpha-beta-pruning to be effective
    :return: The move
    """
    obl_m = obliged_move(board, player)
    if obl_m is not None:
        return obl_m
    move = _minimax_depth_limited_alpha_beta_pruning(board, player, max_depth, eval_f, eval_for_sorting_moves_f,
                                                     -math.inf, math.inf)[0]
    return move

