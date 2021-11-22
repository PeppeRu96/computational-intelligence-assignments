from itertools import combinations
from game_utility import *
from montecarlo_tree_search import *
from minimax_plus_mcts import *

def generate_statistics(players, rounds):
    stats = {p1['id']: {p2['id']: [0, 0] for j, p2 in enumerate(players)} for i, p1 in enumerate(players)}

    comb = len(list(combinations(players, 2)))
    n = 1

    for i, p1 in enumerate(players):
        for j, p2 in enumerate(players):
            if j <= i:
                continue

            pl1 = p1['player']
            pl2 = p2['player']
            pl1.set_player_n(1)
            pl2.set_player_n(-1)
            if 'wrapper_object' in p1 and p1['wrapper_object'] is not None:
                p1['wrapper_object'].player_n = 1


            if 'wrapper_object' in p2 and p2['wrapper_object'] is not None:
                p2['wrapper_object'].player_n = -1

            l = 140
            print(f"Game {n}/{comb}")
            print("------", end="")
            print(" Player 1: ".ljust(15), end="")
            print(f"| {pl1.name} |".ljust(l - 15), end="")
            print("------")
            print("------", end="")
            print(f"vs".center(l), end="")
            print("------")
            print("------", end="")
            print(" Player 2: ".ljust(15), end="")
            print(f"| {pl2.name} |".ljust(l - 15), end="")
            print("------")
            for i in range(rounds):
                print(f"\tRound ({i + 1}/{rounds}).. ", end="")
                start = timeit.default_timer()
                winner = play_connect4(
                    player1=pl1,
                    player2=pl2,
                    show_game_stages=True,
                    show_taken_time=True
                )

                # Reset wrapper object
                if 'wrapper_object' in p1 and p1['wrapper_object'] is not None:
                    old = p1['wrapper_object']
                    if p1['type'] == 'random':
                        f = rollout_f3
                    else:
                        f = rollout_f4

                    new_wrapper_object = MCTS(player_n=old.player_n,
                                              iterations=old.iterations,
                                              max_time=old.max_time,
                                              rollout_f=f,
                                              draw_tree=False,
                                              draw_only_sub_tree=False,
                                              c_param=old.c_param)
                    new_player = Player(new_wrapper_object.play, pl1.name, new_wrapper_object.set_opponent_move)
                    p1['player'] = new_player
                    p1['wrapper_object'] = new_wrapper_object

                if 'wrapper_object' in p2 and p2['wrapper_object'] is not None:
                    old = p2['wrapper_object']
                    if p2['type'] == 'random':
                        f = rollout_f3
                    else:
                        f = rollout_f4

                    new_wrapper_object = MCTS(player_n=old.player_n,
                                              iterations=old.iterations,
                                              max_time=old.max_time,
                                              rollout_f=f,
                                              draw_tree=False,
                                              draw_only_sub_tree=False,
                                              c_param=old.c_param)
                    new_player = Player(new_wrapper_object.play, pl2.name, new_wrapper_object.set_opponent_move)
                    p2['player'] = new_player
                    p2['wrapper_object'] = new_wrapper_object


                end = timeit.default_timer()
                if winner == 1:
                    stats[p1['id']][p2['id']][0] += 1
                    stats[p2['id']][p1['id']][1] += 1
                    print(f"Winner: | {pl1.name} | (player 1) - game duration: {end - start} seconds")
                elif winner == -1:
                    stats[p1['id']][p2['id']][1] += 1
                    stats[p2['id']][p1['id']][0] += 1
                    print(f"Winner: | {pl2.name} | (player 2) - game duration: {end - start} seconds")
                else:
                    print(f"Draw! - game duration: {end - start} seconds")
            print("\n")
            n += 1

    print("Legend")
    for p in players:
        pl = p['player']
        print(f"Id: {p['id'] + 1} - Name: {pl.name}")

    print()
    print("The table shows the number of victories of player 1 (p1) over the number of rounds.")
    print()

    space = 20
    print("".center(space), end="")
    for p in players:
        print(f"{p['id'] + 1} (p2)".center(space), end="")
    print("Overall".center(space), end="")
    print()
    print()
    for i, p1 in enumerate(players):
        print(f"{p1['id'] + 1} (p1)".ljust(space), end="")
        for j, p2 in enumerate(players):
            if j != i:
                print(f"{stats[p1['id']][p2['id']][0]}/{rounds}".center(space), end="")
            else:
                print("".center(space), end="")

        sum = 0
        r = 0
        for j, p2 in enumerate(players):
            if j != i:
                sum += stats[p1['id']][p2['id']][0]
                r += rounds
        print(f"{sum}/{r}".center(space), end="")
        print()
        print()



if __name__ == "__main__":
    # 1. Minimax with Alpha-Beta pruning and random sampling estimation
    max_depth1 = 3
    samples1 = 10
    estimate_f1 = lambda board, turn: random_rollout_estimate(board, turn, samples1)
    estimate_name1 = f"Random rollout estimate with samples: {samples1}"
    sort_move_f1 = lambda board, turn: heuristic_static_magic_square_estimate(board)
    sort_move_name1 = f"Static heuristic with magic square estimate"

    minimax_play1 = lambda board, turn: minimax_depth_limited_alpha_beta_pruning(board=board,
                                                                                player=turn,
                                                                                max_depth=max_depth1,
                                                                                eval_f=estimate_f1,
                                                                                eval_for_sorting_moves_f=sort_move_f1)
    name1 = f"Minimax with Alpha-Beta Pruning (Max depth: {max_depth1}, Estimate: {estimate_name1}, Function for sorting moves: {sort_move_name1})"

    player1 = Player(minimax_play1, name1)
    p1 = {
        'id': 1,
        'player': player1
    }

    # 2. Minimax with Alpha-Beta pruning and Heuristic estimation
    max_depth2 = 6
    estimate_f2 = lambda board, turn: heuristic_static_magic_square_estimate(board)
    estimate_name2 = f"Static heuristic with magic square estimate"
    sort_move_f2 = lambda board, turn: heuristic_static_magic_square_estimate(board)
    sort_move_name2 = f"Static heuristic with magic square estimate"

    minimax_play2 = lambda board, turn: minimax_depth_limited_alpha_beta_pruning(board=board,
                                                                                 player=turn,
                                                                                 max_depth=max_depth2,
                                                                                 eval_f=estimate_f2,
                                                                                 eval_for_sorting_moves_f=sort_move_f2)
    name2 = f"Minimax with Alpha-Beta Pruning (Max depth: {max_depth2}, Estimate: {estimate_name2}, Function for sorting moves: {sort_move_name2})"

    player2 = Player(minimax_play2, name2)
    p2 = {
        'id': 2,
        'player': player2
    }

    # 3. Montecarlo Tree Search with random rollout
    iterations3 = 250
    max_time3 = None
    c_param3 = math.sqrt(2)
    samples3 = 3
    rollout_f3 = lambda board, node: rollout_random_sampling(board, node, samples3)
    rollout_name3 = f"Random sampling estimate (samples: {samples3})"

    end_str3 = f"Iterations: {iterations3}" if iterations3 is not None else f"Max time: {max_time3}"
    montecarlo_name3 = f"Montecarlo Tree Search ({end_str3}, Rollout: {rollout_name3})"

    mcts_wrapper3 = MCTS(player_n=None,
                        iterations=iterations3,
                        max_time=max_time3,
                        rollout_f=rollout_f3,
                        draw_tree=False,
                        draw_only_sub_tree=False,
                        c_param=c_param3)

    mcts_player3 = Player(mcts_wrapper3.play, montecarlo_name3, mcts_wrapper3.set_opponent_move)

    p3 = {
        'id': 3,
        'player': mcts_player3,
        'wrapper_object': mcts_wrapper3,
        'type': 'random'
    }

    # 4. Montecarlo Tree Search with static heuristic rollout
    iterations4 = 1000
    max_time4 = None
    c_param4 = math.sqrt(2)
    rollout_f4 = lambda board, node: rollout_static_heuristic_magic_square(board)
    rollout_name4 = f"Static heuristic magic square estimation)"

    end_str4 = f"Iterations: {iterations4}" if iterations4 is not None else f"Max time: {max_time4}"
    montecarlo_name4 = f"Montecarlo Tree Search ({end_str4}, Rollout: {rollout_name4})"

    mcts_wrapper4 = MCTS(player_n=None,
                        iterations=iterations4,
                        max_time=max_time4,
                        rollout_f=rollout_f4,
                        draw_tree=False,
                        draw_only_sub_tree=False,
                        c_param=c_param4)

    mcts_player4 = Player(mcts_wrapper4.play, montecarlo_name4, mcts_wrapper4.set_opponent_move)

    p4 = {
        'id': 4,
        'player': mcts_player4,
        'wrapper_object': mcts_wrapper4,
        'type': 'heuristic'
    }

    # 5. Minimax plus MCTS with static heuristic evaluation
    max_depth5 = 3
    iterations5 = 30
    c_param5 = math.sqrt(2)

    move_sort_f5 = lambda board, turn: heuristic_static_magic_square_estimate(board)
    move_sort_name5 = f"Static heuristic with magic square estimate"

    rollout_f5 = lambda board, node: rollout_static_heuristic_magic_square(board)
    rollout_name5 = f"Static heuristic with magic square estimate"

    play5 = lambda board, turn: minimax_plus_montecarlo_tree_search(board=board,
                                                                   player=turn,
                                                                   max_depth=max_depth5,
                                                                   sort_move_f=move_sort_f5,
                                                                   iterations=iterations5,
                                                                   rollout_f=rollout_f5,
                                                                   c_param=c_param5)

    name5 = f"Minimax plus MCTS (Max depth: {max_depth5}, Moves sorted by: {move_sort_name5}, MCTS iterations: {iterations5}, Rollout: {rollout_name5})"

    player5 = Player(play5, f"Minimax plus MCTS (Max depth: {max_depth5}, Moves sorted by: {move_sort_name5}, MCTS iterations: {iterations5}, Rollout: {rollout_name5})")

    p5 = {
        'id': 5,
        'player': player5,
    }

    players = [p1, p2, p3, p4, p5]
    #players = [p1, p4]
    generate_statistics(players, 5)