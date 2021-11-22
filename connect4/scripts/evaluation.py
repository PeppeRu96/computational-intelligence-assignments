from connect4 import *
from collections import Counter
import timeit

def check_winner(board):
    """
    Returns if someone has won and who is the winner
    :param board: np array containing the board
    :return: 1 if player 1 has won,
             -1 if player -1 has won,
             0 in case of no winner (not terminal or draw)
    """
    if four_in_a_row(board, 1):
        # Alice won
        return 1
    elif four_in_a_row(board, -1):
        # Bob won
        return -1
    else:
        # Not terminal, 0
        return 0

def obliged_move(board, player):
    """
    If there is a obliged move, it returns it
    :param board:
    :param player: The player who is doing the move
    :return: The obliged move, otherwise None
    """
    # Check winning moves
    for m in valid_moves(board):
        play(board, m, player)
        val = check_winner(board)
        take_back(board, m)
        if val == player:
            return m

    # Check forced (losing otherwise) moves
    for m in valid_moves(board):
        play(board, m, -player)
        val = check_winner(board)
        take_back(board, m)
        if val == -player:
            return m
    return None

# Random rollout estimate
def _random_rollout(board, player):
    p = -player
    while valid_moves(board):
        p = -p
        c = np.random.choice(valid_moves(board))
        play(board, c, p)
        if four_in_a_row(board, p):
            return p

    # draw
    return 0

def random_rollout_estimate(board, player, samples):
    """
    Starting from the current board, do `samples` times a random rollout until the game ends
    and average the winning times to retrieve an estimate.
    :param board:
    :param player: The player which has to do the next move starting from the board passed
    :param samples: The number of times to do the random rollout
    :return: A float in the range [-1, 0) if player -1 has won more times than player 1, whereas (0, 1] otherwise.
    It returns 0 in case the two players have won the smae number of times each.
    """
    cnt = Counter(_random_rollout(np.copy(board), player) for _ in range(samples))
    # cnt[1]: times that player 1 has won
    # cnt[-1]: times that player -1 has won
    return (cnt[1] - cnt[-1]) / samples

# Utility to measure how much time the random rollout takes given an amount of samples, startin from an empty board
def measure_time_random_rollout_estimate(samples, print_result=True):
  board = np.zeros((NUM_COLUMNS, COLUMN_HEIGHT), dtype=np.byte)
  t = timeit.repeat(lambda: random_rollout_estimate(board, 1, samples), number=1, repeat=10)
  t = sum(t) / len(t)
  if print_result:
    print(f"Random rollout estimate took {t} seconds to do {samples} simulations on an empty board")

# Heuristics

# Heuristic - static magic square
HEURISTIC_MAGIC_SQUARE = np.array([
                [3, 4, 5, 5, 4, 3],
                [4, 6, 8, 8, 6, 4],
                [5, 8, 11, 11, 8, 5],
                [7, 10, 13, 13, 10, 7],
                [5, 8, 11, 11, 8, 5],
                [4, 6, 8, 8, 6, 4],
                [3, 4, 5, 5, 4, 3]
], dtype=np.byte)

def display_heuristic_magic_square():
    display(HEURISTIC_MAGIC_SQUARE, "Magic square for static heuristic", use_symbols=False)

def heuristic_static_magic_square_estimate(board):
    """
    It assigns a value to each cell and give that amount to the player which has placed a disk in that cell, if there is one.
    It sums the scores for each cell:
    - For player 1, the scores will be positive
    - For player 2, the scores will be negative
    Finally, it sums everything and average the result by the sum of the magic square
    :param board:
    :return: 0 if player 1 and player 2 have the same scores
             (0, 1] if player 1 has a greater score than player -1
             [-1, 0) if player 1 has a lower score than player -1
    """

    # However, if in this board someone has won, we straight away return 1 or -1
    val = check_winner(board)
    if val != 0:
        return val

    score = int((board * HEURISTIC_MAGIC_SQUARE).sum()) / HEURISTIC_MAGIC_SQUARE.sum()
    return score

