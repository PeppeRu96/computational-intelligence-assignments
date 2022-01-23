from BasePlayer import *

# Wrapper objects to perform async moves
class Move:
    def __init__(self):
        pass

class MovePlay(Move):
    def __init__(self, pos):
        super().__init__()
        self.pos = pos

    def perform(self, client, show=True):
        if show:
            print(f"Going to play card in position {self.pos}")
        client.play(self.pos)

class MoveDiscard(Move):
    def __init__(self, pos):
        super().__init__()
        self.pos = pos

    def perform(self, client, show=True):
        if show:
            print(f"Going to discard card in position {self.pos}")
        client.discard(self.pos)

class MoveHint(Move):
    def __init__(self, fellow_p_name, type, value):
        super().__init__()
        self.fellow_p_name = fellow_p_name
        self.type = type
        self.value = value

    def perform(self, client, show=True):
        if show:
            print(f"Hint to be given to {self.fellow_p_name}: {self.type}: {self.value}")
        client.hint(self.fellow_p_name, self.type, self.value)

# Basic wrapper for any AI, it provides automatic ready and other functionalities
class BaseAI(BasePlayer):
    def __init__(self, show_moves: List[ShowMoves] = None, save_game_states=False):
        super().__init__(show_moves=show_moves, save_game_states=save_game_states)
        self.current_player = None
        self.last_player = None
        self.is_updated = False
        self.ready_to_play = False

    def connection_accepted(self):
        self.client.ready()

    def __update_curr_player(self, curr):
        self.last_player = self.current_player
        self.current_player = curr
        if self.last_player != self.current_player:
            self.ready_to_play = True

    def __update(self, curr):
        self.is_updated = False
        if curr == self.client.player_name:
            self.client.show()

    def __update_all(self, curr):
        self.__update_curr_player(curr)
        self.__update(curr)

    def is_valid_move(self, move):
        if type(move) is MovePlay:
            return True
        elif type(move) is MoveDiscard:
            if self.note_tokens == 0:
                return False
        elif type(move) is MoveHint:
            if self.note_tokens >= 8:
                return False

            p = [p for p in self.players if p.name == move.fellow_p_name][0]
            if move.type == "color":
                c = [c for c in p.hand if c.color == move.value]
                if len(c) < 1:
                    return False
            elif move.type == "value":
                c = [c for c in p.hand if c.value == move.value]
                if len(c) < 1:
                    return False
        return True

    def random_move(self):
        self.ready_to_play = False
        self.is_updated = False

        self.log().info("Making a random move..")

        PLAY = 1
        DISCARD = 2
        HINT = 3
        move_types = [PLAY, DISCARD, HINT]
        if self.note_tokens == 0:
            move_types.remove(DISCARD)
        if self.note_tokens >= 8:
            move_types.remove(HINT)
        if len(move_types) < 2:
            rand_move_type = PLAY
        else:
            rand_move_type = np.random.choice(move_types)

        if rand_move_type == PLAY:
            rand_card_pos = np.random.randint(0, self.my_hand_len)
            MovePlay(rand_card_pos).perform(self.client, show=True)
        elif rand_move_type == DISCARD:
            rand_card_pos = np.random.randint(0, self.my_hand_len)
            MoveDiscard(rand_card_pos).perform(self.client, show=True)
        elif rand_move_type == HINT:
            ps = [p for p in self.players if p.name != self.current_player]
            rand_p = np.random.choice(ps)
            rand_card_pos = np.random.randint(0, len(rand_p.hand))
            type = "color" if np.random.rand() > 0.5 else "value"
            if type == "color":
                value = rand_p.hand[rand_card_pos].color
            elif type == "value":
                value = rand_p.hand[rand_card_pos].value
            MoveHint(rand_p.name, type, value).perform(self.client, show=True)

    # Override event handlers to add functionalities
    def server_start_event(self, data):
        super().server_start_event(data)
        self.__update_all(data.players[0])

    def state_update_event(self, data):
        super().state_update_event(data)
        if self.client.player_name == data.currentPlayer:
            self.is_updated = True

        self.__update_curr_player(data.currentPlayer)

    # Play move ok
    def player_move_event(self, data):
        super().player_move_event(data)
        self.__update_all(data.player)

    # Play move wrong
    def thunder_strike_event(self, data):
        super().thunder_strike_event(data)
        self.__update_all(data.player)

    # Hint
    def hint_event(self, data):
        super().hint_event(data)
        self.__update_all(data.player)

    # Discard
    def discard_event(self, data):
        super().discard_event(data)
        self.__update_all(data.player)

    # Game over
    def game_over_event(self, data):
        super().game_over_event(data)
        self.client.show()

    def reset(self):
        super().reset()
        self.current_player = None
        self.last_player = None
        self.is_updated = False
        self.ready_to_play = False
