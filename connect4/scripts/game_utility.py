from connect4 import *
from evaluation import check_winner
import timeit

class Player:
    def __init__(self, play_f, name, callback_after_opponent_move=None):
        """
        A wrapper to encapsulate a player for conveniently play
        :param play_f: A function (board, player) -> move
        :param name: Player name
        :param callback_after_opponent_move: A function (board, player, opponent_move) -> None
        """
        self.play_f = play_f
        self.name = name
        self.callback_after_opponent_move = callback_after_opponent_move
        self.player_n = None

    def set_player_n(self, n):
        self.player_n = n

# Play wrapper
def play_connect4(player1 : Player, player2 : Player, initial_board=None, turn=None, show_game_stages=True, show_taken_time=False):
    """
    A function that allows to start and fully complete a connect 4 game, given two players
    :param player1: A Player object initialized correctly
    :param player2: A Player object initialized correctly
    :param initial_board: Optionally an initial board
    :param turn: Optionally a turn
    :param show_game_stages: Shows the game stages after each move
    :return: 1 if player 1 wins, -1 if player 2 wins, 0 if draw
    """
    # Initialize the board
    if initial_board is not None:
        board = initial_board
    else:
        board = np.zeros((NUM_COLUMNS, COLUMN_HEIGHT), dtype=np.byte)

    # Assign first turn
    if turn is None:
        turn = 1 if np.random.random() > 0.5 else -1

    # Associate player names to 1 and -1
    player1.set_player_n(1)
    player2.set_player_n(-1)
    player_names = {
        1: player1.name,
        -1: player2.name
    }

    # Associate players with the respective functions
    players = {1: player1.play_f, -1: player2.play_f}

    # Get the initial move number (in case of an initial_board not empty)
    move_cnt = np.abs(board).sum()

    if show_game_stages:
        print(f"Connect4 game starting: {player_names[1]} vs {player_names[-1]}")
        print()
        display(board, "Initial")

    stats_t = {
        1: 0.0,
        -1: 0.0
    }
    stats_m = {
        1: 0,
        -1: 0
    }

    # Game loop
    while True:
        if show_game_stages:
            print(f"Turn of player {player_names[turn]} ({symbols[turn]})")

        if show_taken_time:
            start_t = timeit.default_timer()
        move = players[turn](board, turn)
        if show_taken_time:
            end_t = timeit.default_timer()
            t = end_t - start_t
            stats_t[turn] += t
            stats_m[turn] += 1
            print(f"The player took {end_t - start_t:.1f} seconds to make the move.")
        if move is None:
            break

        # Play the move
        play(board, move, turn)

        # Call the callbacks
        if turn == -1:
            if player1.callback_after_opponent_move is not None:
                player1.callback_after_opponent_move(board, turn, move)
        else:
            if player2.callback_after_opponent_move is not None:
                player2.callback_after_opponent_move(board, turn, move)

        # Display
        move_cnt += 1
        if show_game_stages:
            display(board, f"({move_cnt}) After {player_names[turn]} (symbol: {symbols[turn]}) has placed a disk in column {move +1}")
            print()

        # Check terminal state
        w = check_winner(board)
        if w != 0 or len(valid_moves(board)) == 0:
            break

        # Switch the turn
        turn *= -1

    # Check for the winner
    winner = check_winner(board)

    # Game over
    if show_game_stages:
        print("-------------- GAME OVER --------------")
        display(board, "Final board")
        if winner != 0:
            print(f"The winner is {player_names[winner]}  ({symbols[winner]})! Congratulations!")
        else:
            print(f"Draw!")
        if show_taken_time:
            print(f"{player_names[1]} ({symbols[1]}) took {stats_t[1]/stats_m[1]:.1f} seconds to make a move on average.")
            print(f"{player_names[-1]} ({symbols[-1]}) took {stats_t[-1]/stats_m[-1]:.1f} seconds to make a move on average.")
        print()

    return winner

def build_human_player():
    """
    A simple wrapper to build a human player
    :return:
    """
    def play(board, turn):
        while True:
            possible_moves = valid_moves(board)
            if len(possible_moves) == 0:
                return None

            move = input(f"Insert the column where you want to place your disk {list(np.array(possible_moves) + 1)}: ")
            if move.isdigit():
                move = int(move) - 1
                if move >= 0 and move < NUM_COLUMNS:
                    if move in possible_moves:
                        return move
                    else:
                        print(f"Invalid move. The {move + 1} column is full.")
                else:
                    print(f"You have inserted an invalid column. Column must fall in the range [1-{NUM_COLUMNS}]")
            else:
                print("Invalid input.")

    while True:
        name = input("Please insert a name for the human: ")
        if len(name) > 0:
            break
        print("Invalid name, try again.")
    print(f"{name} welcome to the game!\n")

    human = Player(play, name)
    return human
