#!/usr/bin/env python3

from client_wrapper import Client
from BasePlayer import ShowMoves
from human import Human
from DummyAI import DummyAI
from ZCS_AI import ZCS_AI
from utility import ZCS_DATA_FOLDER
from constants import *
import argparse
import os

p_types = [
    "human",
    "dummy",
    "zcs"
]

# Command-line handling
def get_args():
    parser = argparse.ArgumentParser(description="Learning classifier system (AI) for Hanabi. It will use ZCS_Data folder and will store logs of the game in the logs folder. "
                                                 "Please, type 'exit' when you want to stop the system and wait until it saves the rules. Do not close forcely.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("player_name", type=str, help="The name of the player")
    parser.add_argument("player_type", type=str, default="zcs", nargs='?', help=f"The player type - available types: {p_types}. The dummy player makes only random moves.")
    parser.add_argument("--ip", type=str, default=HOST, help="The ip of the server to connect to")
    parser.add_argument("--port", type=int, default=PORT, help="The port where the server is listening")

    parser.add_argument("--alpha", type=float, default=0.8, help="Defines the importance of the current reward with respect to the previous fitness (momentum)")
    parser.add_argument("--decay", type=int, default=1, help="Defines the decay amount of the activation field for unactive rules, it is applied after each move")
    parser.add_argument("--ga_rate", type=float, default=3, help="Defines the initial genetic algorithm rate, an iteration of GA is performed after ga_rate completed games. "
                                                                 "Watch out, the ga_rate is incremented by 0.01 for each GA iteration.")
    parser.add_argument("--rules", type=int, default=20_000, help="Defines the size of the entire classifier, that is the number of the rules used by the system")
    parser.add_argument("--max_nested_rules", type=int, default=15, help="Defines the maximum number of nested rules that a Compound Rule can be composed of")
    parser.add_argument("--train", default=False, action="store_true", help="Activate the training. Pay a lot of attention! If you enable this and a model_path is not provided, "
                                                                            "by default the model will be saved in ZCS_Data/<player_name>_model.zcs ! If you already have a model trained, "
                                                                            "keep it in a safe position because it may be overwritten!")
    parser.add_argument("--load_from_file", default=False, action="store_true", help="Load the model from file using the provided model_path. If this is done in combination with --train, "
                                                                                     "the model is loaded from the <model_path>, it is then trained, and lastly it is saved again in <model_path>.")
    parser.add_argument("--model_path", type=str, default=None, help="The model path where the model will be loaded from and/or saved to. The file name must have .zcs extension.")
    parser.add_argument("--show_reason", default=False, action="store_true", help="It shows the rule that causes the action performed by the system. Note that it is actually quite verbose.")
    parser.add_argument("--dont_show_move", default=False, action="store_true", help="Prevents the output of the moves performed by the system and other agents.")
    parser.add_argument("--dont_show_status", default=False, action="store_true", help="Prevents the output of the 'show' request after each move.")
    parser.add_argument("--save_game", default=False, action="store_true", help="It will save the game states after each move in a pickle format. It will slow down the process.")

    args = parser.parse_args()
    if args.player_type not in p_types:
        print(f"Invalid player type. Please provide a player type from this set: {p_types}")
        exit(-1)
    args.player_type = args.player_type.lower()

    show_moves = [ShowMoves.SHOW_MOVE, ShowMoves.SHOW_STATUS, ShowMoves.SHOW_EXTENDED_STATUS]
    if args.dont_show_move:
        show_moves.remove(ShowMoves.SHOW_MOVE)
    if args.dont_show_status:
        show_moves.remove(ShowMoves.SHOW_STATUS)
        show_moves.remove(ShowMoves.SHOW_EXTENDED_STATUS)
    args.show_moves = show_moves
    if args.model_path is None:
        args.model_path = os.path.join(ZCS_DATA_FOLDER, f"{args.player_name}_model.zcs")

    return args

if __name__ == "__main__":

    args = get_args()
    p_type = args.player_type
    if p_type == p_types[0]:
        p = Human(save_game_states=args.save_game)
    elif p_type == p_types[1]:
        p = DummyAI(save_game_states=args.save_game)
    elif p_type == p_types[2]:
        p = ZCS_AI(
            alpha=args.alpha,
            decay_amount=args.decay,
            ga_rate=args.ga_rate,
            n_rules=args.rules,
            max_compound_rule_size=args.max_nested_rules,
            train=args.train,
            load_from_file=args.load_from_file,
            model_path=args.model_path,
            show_move_reason=args.show_reason,
            show_moves=args.show_moves,
            save_game_states=args.save_game
        )

    c = Client(args.ip, args.port, args.player_name, p, log_file=True)
    c.connect_and_listen()
