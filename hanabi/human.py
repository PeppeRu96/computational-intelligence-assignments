from BasePlayer import *
from client_wrapper import GameStatus

# Human player
class Human(BasePlayer):
    def __init__(self, show_moves: List[ShowMoves] = None, save_game_states=False):
        super().__init__(show_moves=show_moves, save_game_states=save_game_states)

    # Override console input
    def console_input(self):
        ok = False
        while 1:
            if not ok:
                self.client.print_status()
            else:
                self.logf().info(f"[{self.client.player_name} - {self.client.status.name}]: {command}")

            command = input()

            ok = True
            if command == "exit":
                os._exit(0)
            elif command == "ready":
                if self.client.status is GameStatus.LOBBY:
                    self.client.ready()
                else:
                    print("Command not valid. The game is already running.")
                    ok = False
            elif command == "show":
                if self.client.status == GameStatus.GAME:
                    self.client.show()
                else:
                    print("Command not valid. The game is not yet started.")
                    ok = False
            elif command.split(" ")[0] == "discard":
                if self.client.status == GameStatus.GAME:
                    try:
                        cardStr = command.split(" ")
                        cardOrder = int(cardStr[1])
                        self.client.discard(cardOrder)
                    except:
                        print("Maybe you wanted to type 'discard <num>'?")
                        ok = False
                        continue
                else:
                    print("Command not valid. The game is not yet started.")
            elif command.split(" ")[0] == "play":
                if self.client.status == GameStatus.GAME:
                    try:
                        cardStr = command.split(" ")
                        cardOrder = int(cardStr[1])
                        self.client.play(cardOrder)
                    except:
                        print("Maybe you wanted to type 'play <num>'?")
                        ok = False
                        continue
                else:
                    print("Command not valid. The game is not yet started.")
                    ok = False
            elif command.split(" ")[0] == "hint":
                if self.client.status == GameStatus.GAME:
                    try:
                        sp = command.split(" ")
                        d, t, v = sp[1], sp[2].lower(), sp[3].lower()
                        if t != "colour" and t != "color" and t != "value":
                            print("Error: type can be 'color' or 'value'")
                            ok = False
                            continue
                        if t == "value":
                            v = int(v)
                            if v > 5 or v < 1:
                                print("Error: card values can range from 1 to 5")
                                ok = False
                                continue
                        else:
                            if v not in ["green", "red", "blue", "yellow", "white"]:
                                print("Error: card color can only be green, red, blue, yellow or white")
                                ok = False
                                continue
                        self.client.hint(d, t, v)
                    except:
                        print("Maybe you wanted to type 'hint <destinatary> <type> <value>'?")
                        ok = False
                        continue
                else:
                    print("Command not valid. The game is not yet started.")
                    ok = False
            elif command == "":
                ok = False
            else:
                print(f"Unknown command: {command}")
                ok = False
                continue

    def generic_event(self, data):
        super().generic_event(data)
        if self.client.status == GameStatus.GAME:
            self.client.print_status()
