import GameData
import os
import numpy as np
from typing import List
from enum import Enum
from utility import GameState, HintCard, get_extended_hints, sort_hint_cards, get_fellow_players

card_values_to_counts = {1: 3, 2: 2, 3: 2, 4: 2, 5: 1}

# Useful representation to handle uncertainty in deck/ourselves cards
class StochasticCard:
    colors_to_index = {
        "white": 0,
        "red": 1,
        "blue": 2,
        "yellow": 3,
        "green": 4
    }
    index_to_colors = ["white", "red", "blue", "yellow", "green"]

    def __init__(self, color_probs=None, value_probs=None, joint_prob_matrix=None, card=None, pos=None):
        """
        Colors indexes
        0: White
        1: Red
        2: Blue
        3: Yellow
        4: Green
        Values indexes (id=value-1):
        0: Value 1
        1: Value 2
        2: Value 3
        3: Value 4
        4: Value 5
        :param color_probs: the color probabilities of the card (np.array with size 5)
        :param value_probs: the value probabilities of the card (np.array with size 5
        :param joint_prob_matrix: the joint color,value probabilities of the card (np.array with size 5x5)
        """
        super().__init__()
        self.color_probs = color_probs
        self.value_probs = value_probs
        self.joint_probs = joint_prob_matrix
        self.card = card
        self.pos = pos

    def is_known(self):
        return self.card.color is not None and self.card.value is not None

    def is_unknown(self):
        return self.card.color is None and self.card.value is None

    def adapt_col_val_to_joint(self):
        self.color_probs = self.joint_probs.sum(axis=1)
        self.value_probs = self.joint_probs.sum(axis=0)

    def copy(self):
        color_probs = None
        if self.color_probs is not None:
            color_probs = np.copy(self.color_probs)

        value_probs = None
        if self.value_probs is not None:
            value_probs = np.copy(self.value_probs)

        col_val_probs = None
        if self.joint_probs is not None:
            col_val_probs = np.copy(self.joint_probs)
        c = self.card
        pos = self.pos
        return StochasticCard(color_probs=color_probs, value_probs=value_probs, joint_prob_matrix=col_val_probs, card=c, pos=pos)

    def multiply(self):
        if self.color_probs is not None:
            self.color_probs = self.color_probs * self.color_probs
        if self.value_probs is not None:
            self.value_probs = self.value_probs * self.value_probs
        if self.joint_probs is not None:
            self.joint_probs = self.joint_probs @ self.joint_probs

    @staticmethod
    def make_from_card(c, pos=None):
        col_probs = np.zeros(5)
        val_probs = np.zeros(5)
        col_id = StochasticCard.colors_to_index[c.color]
        val_id = c.value - 1
        col_probs[col_id] = 1
        val_probs[val_id] = 1
        col_val_probs = np.zeros((5, 5))
        col_val_probs[col_id, val_id] = 1
        sc = StochasticCard(color_probs=col_probs, value_probs=val_probs, joint_prob_matrix=col_val_probs, card=c, pos=pos)
        return sc

    @staticmethod
    def make_from_incomplete_card(c, prob_matrix, pos=None):
        col_probs = np.zeros(5)
        val_probs = np.zeros(5)
        if c.color is not None:
            col_id = StochasticCard.colors_to_index[c.color]
            col_probs[col_id] = 1
        if c.value is not None:
            val_id = c.value - 1
            val_probs[val_id] = 1

        col_val_probs = np.zeros((5, 5))
        if c.color is not None and c.value is not None:
            # Complete information
            col_val_probs[col_id, val_id] = 1
        else:
            # Partial information
            col_val_probs = np.copy(prob_matrix)
            if c.color is not None:
                # Conditional probability
                p_color = col_val_probs.sum(axis=1)[col_id]
                col_val_probs /= p_color
                # Now col_val_probs[col_id, i] contains P(V=i | Color = known_col)

                # Zeroing the rest of the matrix
                col_val_probs[:col_id, :] = 0
                col_val_probs[col_id+1:, :] = 0
            elif c.value is not None:
                # Conditional probability
                p_value = col_val_probs.sum(axis=0)[val_id]
                col_val_probs /= p_value
                # Now col_val_probs[i, val_id] contains P(C=i | Value = known_val)

                # Zeroing the rest of the matrix
                col_val_probs[:, :val_id] = 0
                col_val_probs[:, val_id + 1:] = 0

        sc = StochasticCard(color_probs=col_probs, value_probs=val_probs, joint_prob_matrix=col_val_probs, card=c, pos=pos)
        return sc

class ShowMoves(Enum):
    SHOW_MOVE=1
    SHOW_STATUS = 4
    SHOW_EXTENDED_STATUS = 5


class BasePlayer:
    """
    Provides basic storage, update of the user's game state and probability reasoning
    """
    def __init__(self, show_moves: List[ShowMoves] = None, save_game_states=False):
        self.client = None
        # Game status
        self.note_tokens = 0
        self.storm_tokens = 0
        self.table_cards = None
        self.discard_pile = None
        self.players = None
        self.my_hand_len = None
        self.my_hand = []

        if show_moves is None:
            self.show_moves = []
            self.show_moves.append(ShowMoves.SHOW_MOVE)
            self.show_moves.append(ShowMoves.SHOW_STATUS)
            self.show_moves.append(ShowMoves.SHOW_EXTENDED_STATUS)
        else:
            self.show_moves = show_moves
        self.save_games_states = save_game_states

    def save_game(self):
        gs = GameState(p_name=self.client.player_name, players=self.players, discard_pile=self.discard_pile,
                       table=self.table_cards, my_hand=self.my_hand, my_hand_len=self.my_hand_len,
                       note_tokens=self.note_tokens, storm_tokens=self.storm_tokens)
        gs.save()

    def log(self):
        return self.client.logger

    def logf(self):
        return self.client.logger_fonly

    def connection_accepted(self):
        pass

    def console_input(self):
        while 1:
            command = input()
            if command == "exit":
                os._exit(0)

    def update_hints(self, used_card_pos):
        used_hc = [(i, hc) for i, hc in enumerate(self.my_hand) if hc.pos == used_card_pos]
        if len(used_hc) > 0:
            index = used_hc[0][0]
            self.my_hand.pop(index)

        # Shift left other hints positions
        for hc in self.my_hand:
            if hc.pos > used_card_pos:
                hc.pos -= 1

    # def get_extended_hints(self):
    #     hint_cards = sort_hint_cards(self.my_hand)
    #     covered_positions = [hc.pos for hc in hint_cards]
    #     extended_hint_cards = [HintCard.from_pos_to_hint(hint_cards, i) if i in covered_positions else None for i in range(self.my_hand_len)]
    #     return extended_hint_cards

    def get_all_extracted_known_cards(self, players, discard_pile, table, my_hand : List[HintCard]):
        # Gathering all complete information cards on my hand
        my_hand_known = [c for c in my_hand if c.is_known()]

        # Fellow players hands
        ps = get_fellow_players(players, self.client.player_name)
        fellow_cards = [c for p in ps for c in p.hand]

        # Discard pile cards
        discard_cards = discard_pile

        # Table cards
        table_cards = [c for col in table for c in table[col]]


        all_extracted_cards = []
        all_extracted_cards.extend(fellow_cards)
        all_extracted_cards.extend(discard_cards)
        all_extracted_cards.extend(table_cards)
        all_extracted_cards.extend(my_hand_known)
        return all_extracted_cards

    def get_card_counts(self, cards):
        """
        Build a np.array 5x5 containing all the extracted known cards
        (e.g. m[0, 0] contains the number of white ones that have been extracted using the list of cards as parameter)
        """
        counts = np.zeros((5, 5), dtype=np.int)
        for c in cards:
            col_id = StochasticCard.colors_to_index[c.color]
            val_id = c.value - 1
            counts[col_id, val_id] += 1
        return counts

    def get_remaining_card_counts(self, cards):
        """
        Build a np.array 5x5 containing all the yet available count of cards (based on all the known extracted cards)
        (e.g. m[0, 0] contains the number of white ones available using the list of cards as parameter)
        """
        counts = self.get_card_counts(cards)
        b = np.array([3, 2, 2, 2, 1])
        b = b.reshape((1, b.shape[0]))
        counts = b - counts
        return counts

    def remaining_deck_cards(self, all_extracted_cards, my_hand: List[HintCard]) -> int:
        """
        Returns the number of cards available on the deck.
        """
        # First, we retrieve the number of cards completely known in our hand
        # (which has been counter in all_extracted_cards)
        my_hand_known = [c for c in my_hand if c.is_known()]
        return 50 - (len(all_extracted_cards) - len(my_hand_known) + self.client.hand_len)

    def build_stoch_card(self, players, discard_pile, table, my_hand : List[HintCard]):
        """
        Try to use probability reasoning, given a game situation, to retrieve a StochasticCard with the correct
        joint probability matrix. Up to now, it does not take into account the partial hints on our cards.
        For example, we should consider:
        P(C=c, V=v | C1 = c1, V2 = v2, C3 = c3)
        in place of:
        P(C=c, V=v)
        Unfortunately, the conditioning on partial hints seems to not be easy to calculate.
        """
        tot_cards = 50
        # All completely known extracted cards
        # (fellows players, discard pile, table and our completely known hand's cards based on hints)
        all_extracted_cards = self.get_all_extracted_known_cards(players, discard_pile, table, my_hand)

        # This does not correspond to the remaining deck length but to all the unknown and partially unknown cards
        # i.e. the cards in the deck and the unknown cards in our hands (or partially unknown)
        remaining_cards_cnt = tot_cards - len(all_extracted_cards)

        # Compute conditional probability on partial hints
        """
        Not working
        col_counts = remaining_counts.sum(axis=1)
        val_counts = remaining_counts.sum(axis=0)

        Conditioning on partial information
        partial_cards = [c for c in my_hand if c.color is None or c.value is None]

        P(C=c, V=v | C1 = c1, V2 = v2, ...) = P(C=c, V=v, C1=c1, V2 = v2, ...) / P(C1=c1, V2=v2 ...)
        We calculate both numerator and denominator with the composition formula
        First term: P(C=c, V=v, C1 = c1 , V2 = v2 ...) = P(C=c, V=v) * P(C1=c1 | C=c, V=v) * P(V2=v2 | C=c, V=v, C1=c1)...
        c_v = remaining_counts / remaining_cards_cnt

        # Conditioning on the first term (..|C=c, V=v)
        remain_cnt_copy = np.copy(remaining_counts)
        remain_cnt_copy -= 1 # remove the card from the matrix

        conditioned_terms = []
        for pc in partial_cards:
            remain_cnt_tmp = np.copy(remain_cnt_copy)
            col_counts = remain_cnt_tmp.sum(axis=1)
            val_counts = remain_cnt_tmp.sum(axis=0)

            # Remove the card of that color or value because of conditioning
            for col, val in conditioned_terms:
                if col is not None:
                    col_counts[col] -= 1
                elif val is not None:
                    val_counts[val] -= 1

            # Update the total count of cards because of conditioning
            tot_col = col_counts.sum()
            tot_val = val_counts.sum()
        """

        remaining_counts = self.get_remaining_card_counts(all_extracted_cards)
        joint = remaining_counts / remaining_cards_cnt
        color_p = joint.sum(axis=1)
        value_p = joint.sum(axis=0)
        # TODO: Note that this is actually an approximation since we should condition on the partial information of the hints
        sc = StochasticCard(color_probs=color_p, value_probs=value_p, joint_prob_matrix=joint)
        return sc

    # Basic event handlers functionalities (to be called or overridden from inherited classes)
    def generic_event(self, data):
        self.log().info("\n")

    def server_start_event(self, data):
        self.my_hand_len = 5 if len(data.players) <= 3 else 4

        s = "--------------- Game start ---------------\n"
        s += f"Current player: {data.players[0]}"
        self.log().info(s)

    def state_update_event(self, data):
        self.players = data.players
        self.note_tokens = data.usedNoteTokens
        self.storm_tokens = data.usedStormTokens
        self.table_cards = data.tableCards
        self.discard_pile = data.discardPile

        if ShowMoves.SHOW_STATUS in self.show_moves or ShowMoves.SHOW_EXTENDED_STATUS in self.show_moves:
            s = GameState.print_status(self.client.player_name, data.players, data.discardPile, data.tableCards,
                                       self.my_hand, self.my_hand_len, data.usedNoteTokens, data.usedStormTokens,
                                       ShowMoves.SHOW_EXTENDED_STATUS in self.show_moves, current_player=data.currentPlayer)
            self.log().info(s)

        if self.save_games_states:
            self.save_game()

    # Play move ok
    def player_move_event(self, data):
        if data.lastPlayer == self.client.player_name:
            self.update_hints(data.cardHandIndex)

        if ShowMoves.SHOW_MOVE in self.show_moves:
            s = f"Player {data.lastPlayer} played correctly {data.card.toClientString()}. Nice move!\nCurrent player: {data.player}"
            self.log().info(s)

    # Play move wrong
    def thunder_strike_event(self, data):
        if data.lastPlayer == self.client.player_name:
            self.update_hints(data.cardHandIndex)

        if ShowMoves.SHOW_MOVE in self.show_moves:
            s = f"Player {data.lastPlayer} tried to play {data.card.toClientString()}, but it was incorrect!\n"
            s += "OH NO! The Gods are unhappy with you!\n"
            s += f"Current player: {data.player}"
            self.log().info(s)

    # Hint
    def hint_event(self, data):
        if ShowMoves.SHOW_MOVE in self.show_moves:
            value_str = "value" if data.type == "value" else "color"
            s = f"{data.source} gives hint to {data.destination} saying that he/she has cards with {value_str} {data.value} in positions: "
            for i in data.positions:
                s += f"{i}, "
            s = s[:-2]
            s += f"\nCurrent player: {data.player}"
            self.log().info(s)

        if data.destination != self.client.player_name:
            return

        # For each given hint
        for pos in data.positions:
            # For each already stored hint card
            found = False
            for hc in self.my_hand:
                if hc.pos == pos:
                    # Found already an hint with the given pos
                    found = True
                    if data.type == "color":
                        hc.color = data.value
                    elif data.type == "value":
                        hc.value = data.value
                    break

            # If it's not present a hint for the given position..
            if not found:
                if data.type == "color":
                    new_hc = HintCard(color=data.value, value=None, pos=pos)
                elif data.type == "value":
                    new_hc = HintCard(color=None, value=data.value, pos=pos)
                else:
                    raise Exception("Invalid hint type received.")

                self.my_hand.append(new_hc)

    # Discard
    def discard_event(self, data):
        if data.lastPlayer == self.client.player_name:
            self.update_hints(data.cardHandIndex)
            self.my_hand_len = data.handLength

        if ShowMoves.SHOW_MOVE in self.show_moves:
            s = f"Player {data.lastPlayer} discarded {data.card.toClientString()}\n"
            s += f"Current player: {data.player}"
            self.log().info(s)

    # Game over
    def game_over_event(self, data):
        s = f"{data.message}\nScore: {data.score}\n{data.scoreMessage}\n"
        s += "-------------------------------------------------\n\n\n"
        s += "Ready for a new game!"
        self.log().info(s)

        self.reset()

    def reset(self):
        self.note_tokens = 0
        self.storm_tokens = 0
        self.table_cards = None
        self.discard_pile = None
        self.players = None
        self.my_hand = []

    # Event handler called from the client
    def event_handler(self, data):
        if type(data) is GameData.ServerStartGameData:
            self.server_start_event(data)
        elif type(data) is GameData.ServerGameStateData:
            self.state_update_event(data)
        elif type(data) is GameData.ServerPlayerMoveOk:
            self.player_move_event(data)
        elif type(data) is GameData.ServerPlayerThunderStrike:
            self.thunder_strike_event(data)
        elif type(data) is GameData.ServerHintData:
            self.hint_event(data)
        elif type(data) is GameData.ServerActionValid:
            self.discard_event(data)
        elif type(data) is GameData.ServerGameOver:
            self.game_over_event(data)

        self.generic_event(data)
