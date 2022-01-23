from ZCS_Data import *
from utility import *
import pickle

"""
LEARNING CLASSIFIER SYSTEM
@ GIUSEPPE RUGGERI
Suggested way to read this code:
- Fold everything
- Read the high-level comments

This class contains the logic, the ZCS_Data.py contains all the wrapper objects and genetic operators.
"""

# Basic wrapper to save and load the model through pickle
class ZCS_Model:
    def __init__(self, zcs):
        self.rules = zcs.rules
        self.alpha = zcs.alpha
        self.decay_amount = zcs.decay_amount
        self.ga_rate = zcs.ga_rate
        self.n_rules = zcs.n_rules
        self.max_compound_rule_size = zcs.max_compound_rule_size
        self.max_rule_depth = zcs.max_rule_depth


# Main class - it contains all the functionalities for the Learning Classifier System
class ZCS_AI(BaseAI):
    def __init__(self, alpha=0.8, decay_amount=1, ga_rate=1, n_rules=50_000, max_compound_rule_size=15, max_rule_depth=1,
                 train=False, load_from_file=False, model_path=None, show_move_reason=False,
                 show_moves: List[ShowMoves] = None, save_game_states=False):
        """
        :param alpha: defines the importance of the current reward with respect to the last fitness (momentum)
        :param decay_amount: how much to decay the activations of the unactive rules after each AI move
        :param ga_rate: defines the number of games to be played before doing a GA step
        :param n_rules: size of the classifier (number of rules used)
        :param max_compound_rule_size: maximum number of nested rules in Compound rules
        :param train: pay attention, it enters in training mode, dispatching rewards and doing GA
        :param load_from_file: load the model from file
        :param model_path: the model path to load and save the model
        :param show_move_reason: shows the rule which caused the action
        """
        super().__init__(show_moves, save_game_states=save_game_states)
        # LCS rules
        self.rules = None
        self.last_chosen_rule = None
        self.last_move = None
        self.hints_given = []                   # Used for immediate reward (list of HintGiven)

        # ZCS parameters
        self.alpha = alpha                      # Defines the importance of the reward with respect to the trigger strength
        self.decay_amount = decay_amount
        self.ga_rate = ga_rate
        self.n_rules = n_rules
        self.max_compound_rule_size = max_compound_rule_size
        self.max_rule_depth = max_rule_depth

        # Training and showing
        self.train = train
        self.load_from_file = load_from_file
        self.model_path = model_path
        self.show_move_reason = show_move_reason

        if load_from_file:
            model = ZCS_AI.load_model(model_path)
            self.rules = model.rules
            self.alpha = model.alpha
            self.decay_amount = model.decay_amount
            self.ga_rate = model.ga_rate
            self.n_rules = model.n_rules
            self.max_compound_rule_size = model.max_compound_rule_size
            self.max_rule_depth = model.max_rule_depth
            self.reset_rule_counts()
            self.reset_triggers()
        else:
            self.initialize_rules()

    # Initialize the rules randomly
    def initialize_rules(self):
        self.rules = [Rule.make_random(self.max_compound_rule_size) for _ in range(self.n_rules)]

    # Saving and loading functionalities
    def save_model(self):
        path = self.model_path if self.model_path is not None else os.path.join(ZCS_DATA_FOLDER, f"{self.client.player_name}_model.zcs")
        f = open(path, "wb")
        model = ZCS_Model(self)
        pickle.dump(model, f)
        f.close()
        self.log().info(f"Model saved in {path}")

    @staticmethod
    def load_model(model_path):
        f = open(model_path, "rb")
        model = pickle.load(f)
        f.close()
        print(f"Model loaded correctly from path: {model_path}")
        return model

    # Trigging rules functions
    def __trig_card_rule(self, r: CardRule, discard_pile, table):
        triggers = []
        if r.owner_type == "discard_pile":
            discard_triggers = []
            for i, c in enumerate(discard_pile):
                s = r.get_strength(c)
                if s:
                    hc = HintCard(c.color, c.value, i)
                    trigger = TriggeredCardRule(r, s, hc)
                    discard_triggers.append(trigger)
            if not r.rev_num_req_trig and len(discard_triggers) >= r.num_requested_trig:
                triggers.extend(discard_triggers)
            elif r.rev_num_req_trig and len(discard_triggers) < r.num_requested_trig:
                triggers.extend(discard_triggers)
        elif r.owner_type == "table":
            table_cards = [c for col in table for c in table[col]]
            table_triggers = []
            for c in table_cards:
                s = r.get_strength(c)
                if s:
                    hc = HintCard(c.color, c.value)
                    trigger = TriggeredCardRule(r, s, hc)
                    table_triggers.append(trigger)
            if not r.rev_num_req_trig and len(table_triggers) >= r.num_requested_trig:
                triggers.extend(table_triggers)
            elif r.rev_num_req_trig and len(table_triggers) < r.num_requested_trig:
                triggers.extend(table_triggers)
        else:
            raise Exception()

        if len(triggers) > 0:
            r.triggers = triggers
            return triggers
        else:
            return None

    def __trig_fellow_card_rule(self, r: FellowCardRule, players):
        def __trig_fellow(fellow):
            triggers = []
            for c_i, c in enumerate(fellow.hand):
                s = r.get_strength(c)
                if s:
                    hc = HintCard(c.color, c.value, c_i)
                    trigger = TriggeredFellowCardRule(r, s, hc, fellow)
                    triggers.append(trigger)

            if not r.rev_num_req_trig and len(triggers) >= r.num_requested_trig:
                return triggers
            elif r.rev_num_req_trig and len(triggers) < r.num_requested_trig:
                return triggers
            else:
                return None

        if r.fellow_num is not None:
            my_pos = None
            for i, p in enumerate(players):
                if p.name == self.client.player_name:
                    my_pos = i
                    break
            if my_pos is None:
                raise Exception()

            fellow_pos = (my_pos + r.fellow_num) % len(players)
            if fellow_pos == my_pos:
                fellow_pos = (fellow_pos + 1) % len(players)
            fellow = players[fellow_pos]
            triggers = __trig_fellow(fellow)
            if triggers is not None:
                r.triggers = triggers
            return triggers
        else:
            ps = get_fellow_players(players, self.client.player_name)
            all_triggers = []
            for p in ps:
                triggers = __trig_fellow(p)
                if triggers is not None:
                    all_triggers.extend(triggers)
            if len(all_triggers) > 0:
                r.triggers = all_triggers
                return all_triggers
            else:
                return None

    def __trig_stoc_card_rule(self, r: StochasticCardRule, players, discard_pile, table, my_hand):
        triggers = []
        if r.owner_type == "deck":
            deck_triggers = []
            sc = self.build_stoch_card(players, discard_pile, table, my_hand)
            curr_sc = sc.copy()
            for _ in range(r.num_requested_trig):
                s = r.get_strength(curr_sc)
                if s:
                    trigger = TriggeredStochasticCardRule(r, s, curr_sc)
                    deck_triggers.append(trigger)
                    curr_sc = curr_sc.copy()
                    curr_sc.multiply()
                else:
                    break
            if not r.rev_num_req_trig and len(deck_triggers) >= r.num_requested_trig:
                triggers.extend(deck_triggers)
            elif r.rev_num_req_trig and len(deck_triggers) < r.num_requested_trig:
                triggers.extend(deck_triggers)
        elif r.owner_type == "myself":
            scs = []

            # For the cards partially or completely known (from hints)
            # TODO: here we could use conditional probability on partially known cards
            tmp = self.build_stoch_card(players, discard_pile, table, my_hand)
            for hc in my_hand:
                sc = StochasticCard.make_from_incomplete_card(hc, np.copy(tmp.joint_probs), pos=hc.pos)
                scs.append(sc)

            # For the other cards in my hand for which I don't have any hint..
            known_positions = [hc.pos for hc in my_hand if not hc.is_unknown()]
            unknown_positions = [i for i in range(self.my_hand_len) if i not in known_positions]
            tmp = self.build_stoch_card(players, discard_pile, table, my_hand)
            curr_sc = tmp.copy()
            for unk_pos in unknown_positions:
                curr_sc.card = HintCard(pos=unk_pos)
                curr_sc.pos = unk_pos
                scs.append(curr_sc)
                curr_sc = curr_sc.copy()
                curr_sc.multiply()

            myself_triggers = []
            for sc_i, sc in enumerate(scs):
                s = r.get_strength(sc)
                if s:
                    trigger = TriggeredStochasticCardRule(r, s, sc)
                    myself_triggers.append(trigger)
            if not r.rev_num_req_trig and len(myself_triggers) >= r.num_requested_trig:
                triggers.extend(myself_triggers)
            elif r.rev_num_req_trig and len(myself_triggers) < r.num_requested_trig:
                triggers.extend(myself_triggers)
        else:
            raise Exception()

        if len(triggers) > 0:
            r.triggers = triggers
            return triggers
        else:
            return None

    def __trig_token_rule(self, r: TokenRule, note_tokens, storm_tokens):
        s = r.get_strength(note_tokens, storm_tokens)
        if s:
            trigger = TriggeredBasicRule(r, s)
            r.triggers = [trigger]
            return [trigger]
        else:
            return None

    def __trig_num_players_rule(self, r: PlayersNumberRule, players):
        s = r.get_strenght(len(players))
        if s:
            trigger = TriggeredBasicRule(r, s)
            r.triggers = [trigger]
            return [trigger]
        else:
            return None

    def __trig_action(self, a: Action, players, discard_pile, table, my_hand):
        if type(a.rule) is StochasticCardRule:
            self.__trig_stoc_card_rule(a.rule, players, discard_pile, table, my_hand)
        elif type(a.rule) is FellowCardRule:
            self.__trig_fellow_card_rule(a.rule, players)
        else:
            raise Exception()

    def __trig_compound_rule(self, r: CompoundRule, players, discard_pile, table, my_hand, note_tokens, storm_tokens):
        activated = 0
        for r_nested in r.rules:
            if type(r_nested) is CompoundRule:
                triggers = self.__trig_compound_rule(r_nested, players, discard_pile, table, my_hand, note_tokens, storm_tokens)
            elif type(r_nested) is CardRule:
                triggers = self.__trig_card_rule(r_nested, discard_pile, table)
            elif type(r_nested) is FellowCardRule:
                triggers = self.__trig_fellow_card_rule(r_nested, players)
            elif type(r_nested) is StochasticCardRule:
                triggers = self.__trig_stoc_card_rule(r_nested, players, discard_pile, table, my_hand)
            elif type(r_nested) is TokenRule:
                triggers = self.__trig_token_rule(r_nested, note_tokens, storm_tokens)
            elif type(r_nested) is PlayersNumberRule:
                triggers = self.__trig_num_players_rule(r_nested, players)
            else:
                raise Exception()

            if triggers is not None:
                activated += 1
            else:
                return None

        if activated != len(r.rules):
            return None
        else:
            trigger = TriggeredCompoundRule(r)
            r.triggers = [trigger]
            return [trigger]

    def trig_rules(self, players, discard_pile, table, my_hand, note_tokens, storm_tokens):
        for r in self.rules:
            if type(r) is CompoundRule:
                triggers = self.__trig_compound_rule(r, players, discard_pile, table, my_hand, note_tokens, storm_tokens)
            elif type(r) is CardRule:
                triggers = self.__trig_card_rule(r, discard_pile, table)
            elif type(r) is FellowCardRule:
                triggers = self.__trig_fellow_card_rule(r, players)
            elif type(r) is StochasticCardRule:
                triggers = self.__trig_stoc_card_rule(r, players, discard_pile, table, my_hand)
            elif type(r) is TokenRule:
                triggers = self.__trig_token_rule(r, note_tokens, storm_tokens)
            elif type(r) is PlayersNumberRule:
                triggers = self.__trig_num_players_rule(r, players)
            else:
                raise Exception()

            if triggers is not None and r.action is not None:
                self.__trig_action(r.action, players, discard_pile, table, my_hand)
                if len(r.action.rule.triggers) < 1:
                    r.triggers = []

    # Activation update
    def __update_activations(self, r):
        # Store the last activations (used for decaying)
        r.last_activations = r.activations

        if len(r.triggers) > 0:
            r.activations = 1
            r.total_activations += 1

        if type(r) is CompoundRule:
            for nr in r.rules:
                self.__update_activations(nr)

    def update_activations(self):
        for r in self.rules:
            self.__update_activations(r)

    # Decaying the inactive rules
    def __decay(self, r : Rule, amount):
        # Decay the rule if it hasn't been activated in the last turn
        if r.last_activations == r.activations:
            r.activations -= amount
            r.total_activations -= amount
            if r.activations < 0:
                r.activations = 0
            if r.total_activations < 0:
                r.total_activations = 0
        r.last_activations = r.activations

        # Do it for the nested rules
        if type(r) is CompoundRule:
            for nested_r in r.rules:
                self.__decay(nested_r, amount)

    def decay(self, amount):
        for r in self.rules:
            self.__decay(r, amount)

    # Choose action
    def choose_action(self):
        # 1. Sort the rules based on the fitness
        sorted_rules = sorted(self.rules, key=lambda r: r.fitness(self.alpha), reverse=True)

        # 2. Try to perform the action (why could it be invalid? the rule may not cover the other constraints of the game!)
        for r in sorted_rules:
            if r.action is None:
                raise Exception()

            move_type = r.action.move_type
            target_rule = r.action.rule
            # The target rule identifies the target of the encoded action
            # We may have more than one valid target
            # We can sort them by trigger strength and try them
            sorted_triggers = sorted(target_rule.triggers, key=lambda t: t.strength, reverse=True)
            for candidate_t in sorted_triggers:
                if type(candidate_t) is TriggeredStochasticCardRule:
                    pos = candidate_t.stoc_card.pos
                    if move_type == "play":
                        move = MovePlay(pos)
                    elif move_type == "discard":
                        move = MoveDiscard(pos)
                    else:
                        raise Exception()
                elif type(candidate_t) is TriggeredFellowCardRule:
                    hc = candidate_t.hint_card
                    fellow = candidate_t.fellow
                    if move_type == "hint_color":
                        move = MoveHint(fellow.name, "color", hc.color)
                    elif move_type == "hint_value":
                        move = MoveHint(fellow.name, "value", hc.value)
                else:
                    raise Exception()

                if self.is_valid_move(move):
                    return r, candidate_t, move
        return None

    # Make move
    def make_move(self):
        # 1. Trig the rules
        self.trig_rules(self.players, self.discard_pile, self.table_cards, self.my_hand, self.note_tokens, self.storm_tokens)

        # 2. Update the activations of the rules
        self.update_activations()

        # 3. Based on fitness, choose a valid action
        action = self.choose_action()
        if action is None:
            self.last_move = None
            self.last_chosen_rule = None
            self.random_move()
        else:
            selected_rule, candidate_t, selected_move = action[0], action[1], action[2]
            selected_rule.chosen += 1
            selected_rule.total_chosen += 1

            # Used for immediate rewards
            self.last_move = selected_move
            self.last_chosen_rule = selected_rule
            if type(selected_move) is MoveHint:
                self.new_given_hint(selected_rule, candidate_t)

            # Display, if you want, the rule that has been chosen (the highest fitness with a valid action basically)
            if self.show_move_reason:
                s = "Reason for the selected move: it has been chosen the following rule:\n"
                s += f"{selected_rule.toCompleteString()}\n\n"
                self.log().info(s)

        # 4. Reset triggers
        self.reset_triggers()

        # 5. Decay inactive rules
        self.decay(self.decay_amount)

        # 6. Apply the action (send to the server)
        if action is not None:
            self.ready_to_play = False
            self.is_updated = False
            selected_move.perform(self.client, show=True)

    # Assign the final reward to the responsible Rules
    def assign_final_reward(self, reward):
        chosen_rules = [r for r in self.rules if r.chosen > 0]
        for r in chosen_rules:
            r.reward += reward

    # Reset the rule states (episodic)
    def __reset_rule_counts(self, r):
        r.activations = 0
        r.last_activations = 0
        r.chosen = 0

        if type(r) is CompoundRule:
            for nr in r.rules:
                self.__reset_rule_counts(nr)

    def reset_rule_counts(self):
        for r in self.rules:
            self.__reset_rule_counts(r)

    def __reset_triggers(self, r):
        r.triggers = []
        if r.action is not None:
            r.action.rule.triggers = []
        if type(r) is CompoundRule:
            for nr in r.rules:
                self.__reset_triggers(nr)

    def reset_triggers(self):
        for r in self.rules:
            self.__reset_triggers(r)

    # Rules evolution exploiting GA/GP
    def boil_ga(self):
        self.log().info("Boiling rules in the GA pot..\n")
        self.log().info(f"Rule deletion..")
        self.rules = rule_deletion(self.rules, self.log())
        self.log().info(f"Rule specialization..")
        rule_specialization(self.rules, self.log())
        self.log().info(f"Rules generalization..")
        rule_generalization(self.rules, self.log())
        self.log().info(f"Rules merging..")
        self.rules = rule_merging(self.rules, self.max_compound_rule_size, self.log(), quantity_factor=0.1)

    # Override of event handlers
    def console_input(self):
        while 1:
            command = input()
            if command == "exit":
                if self.train:
                    self.save_model()
                    write_rules_on_file(self.rules, self.client.player_name)
                os._exit(0)

    # Game over
    def game_over_event(self, data):
        # TODO: should we update the rewards only during training?
        if self.train:
            reward = score_to_reward(data.score)
            self.assign_final_reward(reward)
            self.log().info(f"Reward {reward} backpropagated after the game ended.")

            ga_count = int(self.ga_rate)
            if ga_count == 0:
                ga_count = 1
            if self.client.game_count % ga_count == 0:
                # Updating the rules..
                self.boil_ga()
                self.ga_rate += 0.1
                self.save_model()

        super().game_over_event(data)

    def generic_event(self, data):
        if self.current_player == self.client.player_name and self.ready_to_play and self.is_updated:
            self.make_move()

    def reset(self):
        super().reset()
        self.last_chosen_rule = None
        self.last_move = None
        self.hints_given = []
        self.reset_rule_counts()
        self.reset_triggers()

    # ------ Give immediate rewards for good and bad playing ------

    def new_given_hint(self, rule, trigger: TriggeredFellowCardRule):
        fellow = trigger.fellow
        hint_type = rule.action.move_type
        if hint_type == "hint_color":
            color = trigger.hint_card.color
            value = None
        elif hint_type == "hint_value":
            color = None
            value = trigger.hint_card.value
        else:
            raise Exception()

        # 1. Search, in the already given hints to the target fellow player, if there are cards already matched by this hint
        target_hints_given = [hg for hg in self.hints_given if hg.fellow.name == fellow.name]
        redundant_hint = True
        for hg in target_hints_given:
            is_new_hint = True
            if color is not None and hg.hint_card.color == color:
                if len(hg.color_rules) > 0:
                    is_new_hint = False
                else:
                    hg.color_rules.append(rule)
            elif value is not None and hg.hint_card.value == value:
                if len(hg.value_rules) > 0:
                    is_new_hint = False
                else:
                    hg.value_rules.append(rule)
            if is_new_hint:
                redundant_hint = False

        # 2. Search, for the cards in the target fellow's hand, for other cards not covered by the previous case
        pos_covered = [hg.hint_card.pos for hg in target_hints_given]
        p = [p for p in self.players if p.name == fellow.name]
        if len(p) < 0:
            return
        p = p[0]
        for c_i, c in enumerate(p.hand):
            if c_i in pos_covered:
                continue

            if color is not None and c.color == color:
                hc = HintCard(c.color, c.value, pos=c_i)
                hg = HintGiven(p, hc)
                hg.color_rules.append(rule)
                self.hints_given.append(hg)
                redundant_hint = False
            elif value is not None and c.value == value:
                hc = HintCard(c.color, c.value, pos=c_i)
                hg = HintGiven(p, hc)
                hg.value_rules.append(rule)
                self.hints_given.append(hg)
                redundant_hint = False

        if redundant_hint:
            reward = REWARD_DUPLICATED_HINT
            rule.reward += reward
            self.log().info(f"Reward {reward} assigned for duplicated hint. BAD!")

    def __remove_given_hints(self, data):
        # Check for remove triggered hints
        target_given_hints = [(i, hg) for i, hg in enumerate(self.hints_given) if hg.fellow.name == data.lastPlayer and hg.hint_card.pos == data.cardHandIndex]
        hints_to_remove = [i for i, hg in target_given_hints]
        if len(target_given_hints) > 1:
            raise Exception()

        if len(target_given_hints) == 1:
            hg_to_remove = target_given_hints[0][1]
            index_to_remove = hints_to_remove[0]
            pos = hg_to_remove.hint_card.pos
            # Update the position of the already given hints
            for hg in self.hints_given:
                if hg.fellow.name == data.lastPlayer and hg.hint_card.pos > pos:
                    hg.hint_card.pos -= 1

            # Remove the hints for the used card
            self.hints_given = [hg for i, hg in enumerate(self.hints_given) if i != index_to_remove]

    # Play move ok
    def player_move_event(self, data):
        super().player_move_event(data)

        if data.lastPlayer == self.client.player_name:
            if self.last_move is not None:
                if type(self.last_move) is not MovePlay or self.last_move.pos != data.cardHandIndex:
                    raise Exception()

                # We played correctly a card, give the reward!
                reward = REWARD_GOOD_PLAY[data.card.value-1]
                self.last_chosen_rule.reward += reward
                self.log().info(f"Reward {reward} given for the good play.")
        else:
            # Check if a fellow player made a good move with our hint to give reward
            for hg in self.hints_given:
                hc = hg.hint_card
                fellow = hg.fellow
                p_name = fellow.name
                if p_name == data.lastPlayer and hc.pos == data.cardHandIndex:
                    reward = REWARD_GOOD_PLAY_AFTER_HINT
                    for r in hg.value_rules:
                        r.reward += reward
                    for r in hg.color_rules:
                        r.reward += reward

                    self.log().info(f"Reward {reward} given for the good hint since {p_name} played correctly ({data.card.value}-{data.card.color}).")
                    break

            self.__remove_given_hints(data)

    # Play move wrong
    def thunder_strike_event(self, data):
        super().thunder_strike_event(data)

        if data.lastPlayer == self.client.player_name:
            if self.last_move is not None:
                if type(self.last_move) is not MovePlay or self.last_move.pos != data.cardHandIndex:
                    raise Exception()

                reward = REWARD_BAD_PLAY[data.card.value-1]

                self.last_chosen_rule.reward += reward
                self.log().info(f"Reward {reward} given for the bad play.")
        else:
            # Check if a fellow player made a bad move with our hint to give reward
            for hg in self.hints_given:
                hc = hg.hint_card
                fellow = hg.fellow
                p_name = fellow.name
                if p_name == data.lastPlayer and hc.pos == data.cardHandIndex:
                    reward = REWARD_BAD_WRONG_PLAY_AFTER_HINT
                    for r in hg.value_rules:
                        r.reward += reward
                    for r in hg.color_rules:
                        r.reward += reward
                    self.log().info(
                        f"Reward {reward} given for the bad hint since {p_name} tried to play it wrongly ({data.card.value}-{data.card.color}).")
                    break

            self.__remove_given_hints(data)

    # Discard
    def __get_discard_info(self, data):
        color = data.card.color
        table_cards = [c for col in self.table_cards for c in self.table_cards[col]]

        table_and_discarded_cards = self.discard_pile
        table_and_discarded_cards.extend(table_cards)
        remaining_card_count = self.get_remaining_card_counts(table_and_discarded_cards)
        col_id = StochasticCard.colors_to_index[color]
        val_id = data.card.value - 1
        remaining_card_count[col_id, val_id] -= 1 # TODO: the self.discard_pile is not updated yet (check?)
        table_col_cards = self.table_cards[color]
        last_table_card_value = len(table_col_cards)

        # Recognize if the color stack is blocked
        # 1. Recognize the smaller value of the missing card that blocks the stack
        blocking_value = None
        for i in range(last_table_card_value + 1, 6):
            if remaining_card_count[col_id, i - 1] == 0:
                blocking_value = i
                break

        return remaining_card_count, last_table_card_value, blocking_value

    def evaluate_discard(self, data, myself=False, rules=None):
        color = data.card.color
        col_id = StochasticCard.colors_to_index[color]
        val_id = data.card.value - 1
        remaining_card_count, last_table_card_value, blocking_value = self.__get_discard_info(data)
        card_str = f"({data.card.value}-{data.card.color})"

        if blocking_value is None:
            # Stack not blocked - Check if the card was unuseful
            if data.card.value <= last_table_card_value:
                # Good discard (unuseful card)
                if myself:
                    reward = REWARD_STACK_NOT_BLOCKED_GOOD_DISCARD_UNUSEFUL_CARD
                    self.log().info(f"Reward {reward} given for the discard. I discarded {card_str} which is actually unuseful.")
                else:
                    reward = REWARD_FELLOW_STACK_NOT_BLOCKED_GOOD_DISCARD_UNUSEFUL_CARD
                    self.log().info(
                        f"Reward {reward} given for the discard. {data.lastPlayer} discarded {card_str} using our hint, it is actually an unuseful card.")
            else:
                # Not very good discard, we did not block the state but we discarded a useful card
                if remaining_card_count[col_id, val_id] == 2:
                    if myself:
                        reward = REWARD_STACK_NOT_BLOCKED_DISCARD_REMAINING_2
                        self.log().info(
                            f"Reward {reward} given for the discard. I discarded {card_str} and now there are two of them left.")
                    else:
                        reward = REWARD_FELLOW_STACK_NOT_BLOCKED_DISCARD_REMAINING_2
                        self.log().info(
                            f"Reward {reward} given for the discard. {data.lastPlayer} discarded {card_str} using our hint and now there are two of them left.")
                elif remaining_card_count[col_id, val_id] == 1:
                    if myself:
                        reward = REWARD_STACK_NOT_BLOCKED_DISCARD_REMAINING_1
                        self.log().info(
                            f"Reward {reward} given for the discard. I discarded {card_str} and now there is the last left.")
                    else:
                        reward = REWARD_FELLOW_STACK_NOT_BLOCKED_DISCARD_REMAINING_1
                        self.log().info(
                            f"Reward {reward} given for the discard. {data.lastPlayer} discarded {card_str} using our hint and now there is the last left.")
                else:
                    raise Exception()
        else:
            if blocking_value == data.card.value:
                # Bad discard, we blocked the stack!
                if myself:
                    reward = REWARD_STACK_BLOCKED_BAD_DISCARD_BLOCK_STACK
                    self.log().info(
                        f"Reward {reward} given for the discard. I discarded {card_str} and i blocked the stack! Bad!")
                else:
                    reward = REWARD_FELLOW_STACK_BLOCKED_BAD_DISCARD_BLOCK_STACK
                    self.log().info(
                        f"Reward {reward} given for the discard. {data.lastPlayer} discarded {card_str} using our hint and he/she blocked the stack! Bad!")
            else:
                if blocking_value < data.card.value:
                    # Good discard, the stack was already blocked
                    if myself:
                        reward = REWARD_STACK_BLOCKED_GOOD_DISCARD_ALREADY_BLOCKED
                        self.log().info(
                            f"Reward {reward} given for the discard. I discarded {card_str} and the {color} stack was already blocked! Good!")
                    else:
                        reward = REWARD_FELLOW_STACK_BLOCKED_GOOD_DISCARD_ALREADY_BLOCKED
                        self.log().info(
                            f"Reward {reward} given for the discard. {data.lastPlayer} discarded {card_str} using our hint and the {color} stack was already blocked! Good!")
                else:
                    # The stack is blocked on a card value greater than our discarded card
                    if myself:
                        reward = REWARD_STACK_BLOCKED_ON_GREATER_CARD
                        self.log().info(
                            f"Reward {reward} given for the discard. I discarded {card_str} and the {color} stack is blocked on the ({blocking_value}-{color})!")
                    else:
                        reward = REWARD_FELLOW_STACK_BLOCKED_ON_GREATER_CARD
                        self.log().info(
                            f"Reward {reward} given for the discard. {data.lastPlayer} discarded {card_str} using our hint and the {color} stack is blocked on the ({blocking_value}-{color})!")

        # Assign reward
        if myself:
            self.last_chosen_rule.reward += reward
        else:
            for r in rules:
                r.reward += reward

    def evaluate_fellow_discard(self, data):
        # Check if a fellow player made a good or bad discard using our hint
        for hg in self.hints_given:
            hc = hg.hint_card
            fellow = hg.fellow
            p_name = fellow.name
            if p_name == data.lastPlayer and hc.pos == data.cardHandIndex:
                rules = hg.color_rules
                rules.extend(hg.value_rules)
                self.evaluate_discard(data, myself=False, rules=rules)
                break

    def discard_event(self, data):
        super().discard_event(data)

        if data.lastPlayer == self.client.player_name:
            if self.last_move is not None:
                if type(self.last_move) is not MoveDiscard or self.last_move.pos != data.cardHandIndex:
                    raise Exception()

                # Check if we discard a good or bad card
                self.evaluate_discard(data, myself=True)
        else:
            # Check if a player discarded a good card after our hint
            self.evaluate_fellow_discard(data)
            self.__remove_given_hints(data)
