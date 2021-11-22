import numpy as np

# Static initialization of the game constraints and size
NUM_COLUMNS = 7
COLUMN_HEIGHT = 6
FOUR = 4
# NB: Connect 4 "columns" are actually NumPy "rows"

# Basic functions
def valid_moves(board):
    """Returns columns where a disc may be played"""
    return [n for n in range(NUM_COLUMNS) if board[n, COLUMN_HEIGHT - 1] == 0]


def play(board, column, player):
    """Updates `board` as `player` drops a disc in `column`"""
    (index,) = next((i for i, v in np.ndenumerate(board[column]) if v == 0))
    board[column, index] = player


def take_back(board, column):
    """Updates `board` removing top disc from `column`"""
    (index,) = [i for i, v in np.ndenumerate(board[column]) if v != 0][-1]
    board[column, index] = 0


def four_in_a_row(board, player):
    """Checks if `player` has a 4-piece line"""
    return (
        any(
            all(board[c, r] == player)
            for c in range(NUM_COLUMNS)
            for r in (list(range(n, n + FOUR)) for n in range(COLUMN_HEIGHT - FOUR + 1))
        )
        or any(
            all(board[c, r] == player)
            for r in range(COLUMN_HEIGHT)
            for c in (list(range(n, n + FOUR)) for n in range(NUM_COLUMNS - FOUR + 1))
        )
        or any(
            np.all(board[diag] == player)
            for diag in (
                (range(ro, ro + FOUR), range(co, co + FOUR))
                for ro in range(0, NUM_COLUMNS - FOUR + 1)
                for co in range(0, COLUMN_HEIGHT - FOUR + 1)
            )
        )
        or any(
            np.all(board[diag] == player)
            for diag in (
                (range(ro, ro + FOUR), range(co + FOUR - 1, co - 1, -1))
                for ro in range(0, NUM_COLUMNS - FOUR + 1)
                for co in range(0, COLUMN_HEIGHT - FOUR + 1)
            )
        )
    )


def colored(r, g, b, text):
    return "\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r, g, b, text)

# Display board utility
symbols_basic = {
        1: f" {colored(255, 0, 0, 'X')} ",
        -1: f" {colored(0, 0, 255, 'S')} ",
        0:  '0'
}

symbols_with_colors = {
        1: f" {colored(255, 0, 0, 'X')} ",
        -1: f" {colored(0, 0, 255, 'S')} ",
        0:  '0'
}

symbols = symbols_with_colors

def display(board, name="", use_symbols=True):
    print(f"{name}:")
    for _ in range(NUM_COLUMNS):
        print("-".center(4, '-'), end="")
    print('---')

    for i in range(COLUMN_HEIGHT - 1, -1, -1):
        print("| ", end="")
        for j in range(NUM_COLUMNS):
            s = symbols[board[j, i]] if use_symbols else board[j, i]
            print(f"{s}".center(4), end="")
        print("|")

    for _ in range(NUM_COLUMNS):
        print("-".center(4, '-'), end="")
    print('---')