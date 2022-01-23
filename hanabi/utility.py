import os
import pickle
from typing import List

SCRIPT_PATH = os.path.dirname(__file__)
GAME_STATES_DATA_FOLDER = os.path.join(SCRIPT_PATH, "game_states")
GS_FNAME = "game_state_1.gs"

ZCS_DATA_FOLDER = os.path.join(SCRIPT_PATH, "ZCS_Data")

if not os.path.exists(GAME_STATES_DATA_FOLDER):
    os.mkdir(GAME_STATES_DATA_FOLDER)

if not os.path.exists(ZCS_DATA_FOLDER):
    os.mkdir(ZCS_DATA_FOLDER)

class HintCard:
    def __init__(self, color=None, value=None, pos=None):
        self.color = color
        self.value = value
        self.pos = pos

    def is_known(self):
        return self.color is not None and self.value is not None

    def is_unknown(self):
        return self.color is None and self.value is None

    def toString(self):
        if self.is_known():
            return f"Card ({self.value}-{self.color})"

        color_str = self.color if self.color is not None else "unknown"
        value_str = self.value if self.value is not None else "unknown"
        return f"Card - value: {value_str}, color: {color_str}"

    @staticmethod
    def from_pos_to_hint(hint_cards, pos):
        a = [hc for hc in hint_cards if hc.pos == pos]
        if len(a) > 0:
            return a[0]
        return None


class GameState:
    def __init__(self, p_name, players, discard_pile, table, my_hand, my_hand_len, note_tokens, storm_tokens):
        self.p_name = p_name
        self.players = players
        self.discard_pile = discard_pile
        self.table = table
        self.my_hand = my_hand
        self.my_hand_len = my_hand_len
        self.note_tokens = note_tokens
        self.storm_tokens = storm_tokens

    def save(self):
        path = incremental_path(GAME_STATES_DATA_FOLDER, f"{self.p_name}_{GS_FNAME}")
        f = open(path, "wb")
        pickle.dump(self, f)
        f.close()

    @staticmethod
    def load(path):
        f = open(path, "rb")
        gs = pickle.load(f)
        f.close()
        return gs

    @staticmethod
    def print_status(my_player, players, discard_pile, table, my_hand, my_hand_len, note_tokens,
                     storm_tokens, extended_status, current_player=""):
        extended_hints = get_extended_hints(my_hand, my_hand_len)
        s = "Player hands: \n"
        if not extended_status:
            for p in players:
                s += f"{p.toClientString()}\n"
        else:
            extended_hint_cards = extended_hints
            for p in players:
                if p.name == my_player:
                    c = "[ \n\t"
                    for card in extended_hint_cards:
                        if card is not None:
                            c += "\t" + card.toString() + " \n\t"
                        else:
                            c += "\tUnknown\n\t"
                    c += " ]"
                    k1 = "{"
                    k2 = "}"
                    s += f"Player {p.name} (myself) {k1} \n\tcards: {c}\n{k2}\n"
                else:
                    s += f"{p.toClientString()}\n"

        s += "Table cards: \n"
        for col in table:
            col_str = f"{col}:"
            s += f"{col_str:<20} [ "
            for c in table[col]:
                s += f"{c.value:3}   "
            s += "]\n\n"

        s += "Discard pile: "
        for c in discard_pile:
            s += f"({c.value}-{c.color}), "
        s = s[:-2]
        s += "\n"
        s += f"Note tokens available: {8 - note_tokens}/8\n"
        s += f"Storm tokens available: {3 - storm_tokens}/3\n"
        s += f"Current player: {current_player}\n"
        return s

def get_fellow_players(all_players, my_player_name):
    return [p for p in all_players if p.name != my_player_name]

def get_extended_hints(my_hand, my_hand_len):
    hint_cards = sort_hint_cards(my_hand)
    covered_positions = [hc.pos for hc in hint_cards]
    extended_hint_cards = [HintCard.from_pos_to_hint(hint_cards, i) if i in covered_positions else None for i in
                           range(my_hand_len)]
    return extended_hint_cards

def sort_hint_cards(hint_cards : List[HintCard]):
    return sorted(hint_cards, key=lambda c: c.pos)

def fetch_game_states():
    file_names = [f for f in os.listdir(GAME_STATES_DATA_FOLDER) if os.path.isfile(os.path.join(GAME_STATES_DATA_FOLDER, f))]
    for fn in file_names:
        yield GameState.load(os.path.join(GAME_STATES_DATA_FOLDER, fn))

def incremental_path(basepath, filename):
    complete_path = os.path.join(basepath, filename)
    if not os.path.exists(complete_path):
        return complete_path

    target_f = filename.split(".")
    target_fname = target_f[-2]
    target_fext = target_f[-1]
    target_f = target_fname.split("_")
    target_fbasename = target_f[:-1]
    target_fbasename = "_".join(target_fbasename)
    target_fid = int(target_f[-1])

    onlyfiles = [f for f in os.listdir(basepath) if os.path.isfile(os.path.join(basepath, f))]
    for f in onlyfiles:
        f = f.split(".")
        fname = f[-2]
        fext = f[-1]
        f = fname.split("_")
        fbasename = f[:-1]
        fbasename = "_".join(fbasename)
        if fbasename == target_fbasename:
            fid = int(f[-1])
            if fid >= target_fid:
                target_fid = fid + 1

    newfname = "%s_%d.%s" % (target_fbasename, target_fid, target_fext)

    return os.path.join(basepath, newfname)

def write_rules_on_file(rules, name):
    f = open(os.path.join(ZCS_DATA_FOLDER, f"{name}_rules.txt"), "w")
    for i, r in enumerate(rules):
            f.write(f"{i}: {r.toString()}")

    f.close()