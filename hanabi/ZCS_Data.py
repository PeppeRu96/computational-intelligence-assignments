import math
from BaseAI import *
from game import Card
from scipy.special import softmax

# Patterns
class Pattern:
    pass

class CardPattern(Pattern):
    color_types = StochasticCard.index_to_colors
    def __init__(self, color, value, joint: bool, constr_type, constr_reversed: bool):
        super().__init__()
        self.color = color
        self.value = value
        self.joint = joint
        self.constr_type = constr_type
        self.constr_reversed = constr_reversed

        # Self-adaptation parameters (discrete in this case)
        self.color_mutation = DiscreteMutation(initial_prob=0.7, learning_rate=10)
        self.value_mutation = DiscreteMutation(initial_prob=0.7, learning_rate=10)

    def match(self, card: Card):
        if self.color is not None:
            if not self.constr_reversed:
                if card.color != self.color:
                    return False
            else:
                if card.color == self.color:
                    return False

        if self.value is not None:
            if not self.constr_reversed:
                if card.value != self.value:
                    return False
            else:
                if card.value == self.value:
                    return False

        return True

    def is_similar(self, other):
        return self.constr_type == other.constr_type and self.constr_reversed == other.constr_reversed

    def creep_mutation(self):
        constr_type = self.constr_type
        color_mutation = self.color_mutation
        value_mutation = self.value_mutation

        def mutate_color():
            color_mutation.mutate()
            self.color = color_mutation.switch_value(self.color, CardPattern.color_types)

        def mutate_value():
            value_mutation.mutate()
            self.value = value_mutation.switch_value(self.value, list(range(1, 6)))

        if constr_type == "joint":
            mutate_color()
            mutate_value()
        else:
            if constr_type == "color_only":
                mutate_color()
            else:
                mutate_value()

    def crossover(self, other):
        if np.random.rand() > 0.5:
            return self.average_crossover(other)
        return self.uniform_crossover(other)

    def average_crossover(self, other):
        def xover_get_color():
            v1_col_id = StochasticCard.colors_to_index[self.color] if self.color is not None else None
            v2_col_id = StochasticCard.colors_to_index[other.color] if other.color is not None else None
            xover_col_id = np.abs(v1_col_id - v2_col_id) // 2
            return StochasticCard.index_to_colors[xover_col_id]

        def xover_get_value():
            v = np.abs(self.value - other.value) // 2
            v = np.clip(v, 1, 5)
            return v

        xover_color, xover_value, xover_joint = None, None, False

        constr_type = self.constr_type
        if constr_type == "joint":
            xover_color = xover_get_color()
            xover_value = xover_get_value()
            xover_joint = True
        else:
            if constr_type == "color_only":
                xover_color = xover_get_color()
            else:
                xover_value = xover_get_value()

        return CardPattern(xover_color, xover_value, xover_joint, constr_type, self.constr_reversed)

    def uniform_crossover(self, other):
        if np.random.rand() > 0.5:
            start = self
            end = other
        else:
            start = other
            end = self

        xover_color, xover_value, xover_joint = None, None, False

        constr_type = self.constr_type
        if constr_type == "joint":
            xover_color = start.color
            xover_value = end.value
            xover_joint = True
        else:
            if constr_type == "color_only":
                xover_color = start.color
            else:
                xover_value = end.value

        return CardPattern(xover_color, xover_value, xover_joint, constr_type, self.constr_reversed)

    @staticmethod
    def make_random():
        color = np.random.choice(CardPattern.color_types)
        value = np.random.randint(1, 6)

        if np.random.rand() > 0.6:
            joint = True
            constr_type = "joint"
        else:
            joint = False
            if np.random.rand() > 0.5:
                value = None
                constr_type = "color_only"
            else:
                color = None
                constr_type = "value_only"

        constr_reversed = np.random.rand() > 0.95

        return CardPattern(color, value, joint, constr_type, constr_reversed)

class StochasticCardPattern(Pattern):
    joint_constr_types = [None, "one_card", "all_cards"]
    joint_constr_types_p = [0.7, 0.2, 0.1]
    def __init__(self, color_id: int, value_id:int, joint_constr, constr_type, color_probs, value_probs, joint_probs, constr_reversed: bool):
        super().__init__()
        self.color_id = color_id
        self.value_id = value_id
        self.joint_constr = joint_constr
        self.constr_type = constr_type
        self.color_probs = color_probs
        self.value_probs = value_probs
        self.joint_probs = joint_probs
        self.constr_reversed = constr_reversed

        # Self-adaptation parameters
        self.color_probs_mutation = ContinuousMutation(initial_sigma=1.0, learning_rate=10)
        self.value_probs_mutation = ContinuousMutation(initial_sigma=1.0, learning_rate=10)
        self.joint_probs_mutation = ContinuousMutation(initial_sigma=1.0, learning_rate=10)

    def get_strength(self, stoc_card: StochasticCard):
        joint_constr = self.joint_constr
        constr_type = self.constr_type
        sc = stoc_card
        color_id = self.color_id
        value_id = self.value_id
        if joint_constr is None:
            if constr_type == "one_col":
                candidate_p = sc.color_probs[color_id]
                constr_p = self.color_probs[color_id]
                if not self.constr_reversed:
                    if candidate_p >= constr_p:
                        return candidate_p - constr_p
                    else:
                        return 0
                else:
                    if candidate_p < constr_p:
                        return constr_p - candidate_p
                    else:
                        return 0
            elif constr_type == "one_val":
                candidate_p = sc.value_probs[value_id]
                constr_p = self.value_probs[value_id]
                if not self.constr_reversed:
                    if candidate_p >= constr_p:
                        return candidate_p - constr_p
                    else:
                        return 0
                else:
                    if candidate_p < constr_p:
                        return constr_p - candidate_p
                    else:
                        return 0
            elif constr_type == "one_col_one_val":
                candidate_color_p = sc.color_probs[color_id]
                constr_color_p = self.color_probs[color_id]
                candidate_value_p = sc.value_probs[value_id]
                constr_value_p = self.value_probs[value_id]
                if not self.constr_reversed:
                    if candidate_color_p >= constr_color_p and candidate_value_p >= constr_value_p:
                        s = candidate_color_p - constr_color_p
                        s += candidate_value_p - constr_value_p
                        s /= 2
                        return s
                    else:
                        return 0
                else:
                    if candidate_color_p < constr_color_p and candidate_value_p < constr_value_p:
                        s = constr_color_p - candidate_color_p
                        s += constr_value_p - candidate_value_p
                        s /= 2
                        return s
                    else:
                        return 0
            elif constr_type == "all_colors":
                if not self.constr_reversed:
                    s = 0
                    for i in range(5):
                        cand_color_p = sc.color_probs[i]
                        constr_color_p = self.color_probs[i]
                        if cand_color_p >= constr_color_p:
                            s += cand_color_p - constr_color_p
                        else:
                            return 0
                    return s
                else:
                    s = 0
                    for i in range(5):
                        cand_color_p = sc.color_probs[i]
                        constr_color_p = self.color_probs[i]
                        if cand_color_p < constr_color_p:
                            s += constr_color_p - cand_color_p
                        else:
                            return 0
                    return s
            elif constr_type == "all_values":
                if not self.constr_reversed:
                    s = 0
                    for i in range(5):
                        cand_value_p = sc.value_probs[i]
                        constr_value_p = self.value_probs[i]
                        if cand_value_p >= constr_value_p:
                            s += cand_value_p - constr_value_p
                        else:
                            return 0
                    return s
                else:
                    s = 0
                    for i in range(5):
                        cand_value_p = sc.value_probs[i]
                        constr_value_p = self.value_probs[i]
                        if cand_value_p < constr_value_p:
                            s += constr_value_p - cand_value_p
                        else:
                            return 0
                    return s
            elif constr_type == "all_colors_all_values":
                if not self.constr_reversed:
                    s = 0
                    for i in range(5):
                        cand_color_p = sc.color_probs[i]
                        constr_color_p = self.color_probs[i]
                        if cand_color_p >= constr_color_p:
                            s += cand_color_p - constr_color_p
                        else:
                            return 0
                    for i in range(5):
                        cand_value_p = sc.value_probs[i]
                        constr_value_p = self.value_probs[i]
                        if cand_value_p >= constr_value_p:
                            s += cand_value_p - constr_value_p
                        else:
                            return 0
                    s /= 2
                    return s
                else:
                    s = 0
                    for i in range(5):
                        cand_color_p = sc.color_probs[i]
                        constr_color_p = self.color_probs[i]
                        if cand_color_p < constr_color_p:
                            s += constr_color_p - cand_color_p
                        else:
                            return 0
                    for i in range(5):
                        cand_value_p = sc.value_probs[i]
                        constr_value_p = self.value_probs[i]
                        if cand_value_p < constr_value_p:
                            s += constr_value_p - cand_value_p
                        else:
                            return 0
                    s /= 2
                    return s
        elif joint_constr == "one_card":
            cand_p = sc.joint_probs[color_id, value_id]
            constr_p = self.joint_probs[color_id, value_id]
            if not self.constr_reversed:
                if cand_p >= constr_p:
                    return cand_p - constr_p
                else:
                    return 0
            else:
                if cand_p < constr_p:
                    return constr_p - cand_p
                else:
                    return 0
        elif joint_constr == "all_cards":
            if not self.constr_reversed:
                if (sc.joint_probs < self.joint_probs).sum() > 0:
                    return 0
                return (sc.joint_probs - self.joint_probs).sum()
            else:
                if (sc.joint_probs >= self.joint_probs).sum() > 0:
                    return 0
                return (self.joint_probs - sc.joint_probs).sum()
        else:
            raise Exception()

    def is_similar(self, other):
        return self.constr_type == other.constr_type and self.joint_constr == other.joint_constr and self.constr_reversed == other.constr_reversed

    def creep_mutation(self):
        joint_constr = self.joint_constr
        constr_type = self.constr_type
        color_id = self.color_id
        value_id = self.value_id
        color_probs_mutation = self.color_probs_mutation
        value_probs_mutation = self.value_probs_mutation
        joint_probs_mutation = self.joint_probs_mutation

        if joint_constr is None:
            if constr_type == "one_col":
                color_probs_mutation.mutate()
                sigma_col = color_probs_mutation.sigma
                self.color_probs[color_id] += np.random.normal(0, np.abs(sigma_col))
            elif constr_type == "one_val":
                value_probs_mutation.mutate()
                sigma_val = value_probs_mutation.sigma
                self.value_probs[value_id] += np.random.normal(0, np.abs(sigma_val))
            elif constr_type == "one_col_one_val":
                color_probs_mutation.mutate()
                sigma_col = color_probs_mutation.sigma
                self.color_probs[color_id] += np.random.normal(0, np.abs(sigma_col))

                value_probs_mutation.mutate()
                sigma_val = value_probs_mutation.sigma
                self.value_probs[value_id] += np.random.normal(0, np.abs(sigma_val))
            elif constr_type == "all_colors":
                color_probs_mutation.mutate()
                sigma_col  =color_probs_mutation.sigma
                self.color_probs += np.random.normal(0, np.abs(sigma_col), size=(5))
            elif constr_type == "all_values":
                value_probs_mutation.mutate()
                sigma_val = value_probs_mutation.sigma
                self.value_probs += np.random.normal(0, np.abs(sigma_val), size=(5))
            elif constr_type == "all_colors_all_values":
                color_probs_mutation.mutate()
                sigma_col = color_probs_mutation.sigma
                self.color_probs += np.random.normal(0, np.abs(sigma_col), size=(5))

                value_probs_mutation.mutate()
                sigma_val = value_probs_mutation.sigma
                self.value_probs += np.random.normal(0, np.abs(sigma_val), size=(5))
        else:
            joint_probs_mutation.mutate()
            sigma_joint = joint_probs_mutation.sigma
            if joint_constr == "one_card":
                self.joint_probs[color_id, value_id] += np.random.normal(0, np.abs(sigma_joint))
            elif joint_constr == "all_cards":
                self.joint_probs += np.random.normal(0, np.abs(sigma_joint), size=(5, 5))
            else:
                raise Exception()

        self.color_probs = softmax(self.color_probs)
        self.value_probs = softmax(self.value_probs)
        self.joint_probs = softmax(self.joint_probs)

    def crossover(self, other):
        if np.random.rand() > 0.5:
            return self.average_crossover(other)
        return self.uniform_crossover(other)

    def average_crossover(self, other):
        joint_constr = self.joint_constr

        v1_color_probs = np.copy(self.color_probs)
        v1_value_probs = np.copy(self.value_probs)
        v1_joint_probs = np.copy(self.joint_probs)

        v2_color_probs = np.copy(other.color_probs)
        v2_value_probs = np.copy(other.value_probs)
        v2_joint_probs = np.copy(other.joint_probs)

        xover_color_probs = np.zeros(5)
        xover_value_probs = np.zeros(5)
        xover_joint_probs = np.zeros((5, 5))

        if joint_constr is None:
            xover_color_probs = np.abs(v1_color_probs - v2_color_probs) / 2
            xover_value_probs = np.abs(v1_value_probs - v2_value_probs) / 2
        else:
            xover_joint_probs = np.abs(v1_joint_probs - v2_joint_probs) / 2

        xover_color_probs = softmax(xover_color_probs)
        xover_value_probs = softmax(xover_value_probs)
        xover_joint_probs = softmax(xover_joint_probs)

        return StochasticCardPattern(self.color_id, self.value_id, joint_constr, self.constr_type, xover_color_probs,
                                     xover_value_probs, xover_joint_probs, self.constr_reversed)

    def uniform_crossover(self, other):
        joint_constr = self.joint_constr
        constr_type = self.constr_type
        color_id = self.color_id
        value_id = self.value_id

        if np.random.rand() > 0.5:
            start = self
            end = other
        else:
            start = other
            end = self

        def nparray_uniform(a: np.ndarray, b: np.ndarray):
            c = np.copy(a)
            v = [a, b]
            curr = 0
            if a.ndim == 2:
                for i in range(a.shape[0]):
                    for j in range(a.shape[1]):
                        c[i, j] = v[curr][i, j]
                        curr = 1 if curr == 0 else 0
            elif a.ndim == 1:
                for i in range(a.shape[0]):
                    c[i] = v[curr][i]
            else:
                raise Exception("Not implemented")

            return c

        v1_color_probs = np.copy(start.color_probs)
        v1_value_probs = np.copy(start.value_probs)
        v1_joint_probs = np.copy(start.joint_probs)

        v2_color_probs = np.copy(end.color_probs)
        v2_value_probs = np.copy(end.value_probs)
        v2_joint_probs = np.copy(end.joint_probs)

        xover_color_probs = np.zeros(5)
        xover_value_probs = np.zeros(5)
        xover_joint_probs = np.zeros((5, 5))

        if joint_constr is None:
            if constr_type == "one_col":
                xover_color_probs[color_id] = v1_color_probs[color_id]
            elif constr_type == "one_val":
                xover_value_probs[value_id] = v2_value_probs[value_id]
            elif constr_type == "one_col_one_val":
                xover_color_probs[color_id] = v1_color_probs[color_id]
                xover_value_probs[value_id] = v2_value_probs[value_id]
            elif constr_type == "all_colors":
                xover_color_probs = nparray_uniform(v1_color_probs, v2_color_probs)
            elif constr_type == "all_values":
                xover_value_probs = nparray_uniform(v1_value_probs, v2_value_probs)
            elif constr_type == "all_colors_all_values":
                xover_color_probs = nparray_uniform(v1_color_probs, v2_color_probs)
                xover_value_probs = nparray_uniform(v1_value_probs, v2_value_probs)
        else:
            if joint_constr == "one_card":
                xover_joint_probs[color_id, value_id] = v1_joint_probs[color_id, value_id]
            elif joint_constr == "all_cards":
                xover_joint_probs = nparray_uniform(v1_joint_probs, v2_joint_probs)
            else:
                raise Exception()

        xover_color_probs = softmax(xover_color_probs)
        xover_value_probs = softmax(xover_value_probs)
        xover_joint_probs = softmax(xover_joint_probs)

        return StochasticCardPattern(color_id, value_id, joint_constr, constr_type, xover_color_probs, xover_value_probs,
                                     xover_joint_probs, self.constr_reversed)

    @staticmethod
    def make_random():
        color_id = None
        value_id = None
        joint_constr = None
        constr_type = None
        color_probs = np.zeros(5)
        value_probs = np.zeros(5)
        joint_probs = np.zeros((5, 5))

        joint_constr = np.random.choice(StochasticCardPattern.joint_constr_types, p=StochasticCardPattern.joint_constr_types_p)
        if joint_constr is None:
            if np.random.rand() > 0.2:
                if np.random.rand() > 0.3:
                    if np.random.rand() > 0.5:
                        # One color
                        color_id = np.random.randint(0, 5)
                        color_p = np.random.rand()
                        color_probs[color_id] = color_p
                        constr_type = "one_col"
                    else:
                        # One value
                        value_id = np.random.randint(0, 5)
                        value_p = np.random.rand()
                        value_probs[value_id] = value_p
                        constr_type = "one_val"
                else:
                    # One color and one value
                    color_id = np.random.randint(0, 5)
                    color_p = np.random.rand()
                    color_probs[color_id] = color_p
                    value_id = np.random.randint(0, 5)
                    value_p = np.random.rand()
                    value_probs[value_id] = value_p
                    constr_type = "one_col_one_val"
            else:
                if np.random.rand() > 0.2:
                    if np.random.rand() > 0.5:
                        # All colors constraints
                        color_probs = np.random.rand(5)
                        constr_type = "all_colors"
                    else:
                        # All values constraints
                        value_probs = np.random.rand(5)
                        constr_type = "all_values"
                else:
                    # Both colors and values constraints
                    color_probs = np.random.rand(5)
                    value_probs = np.random.rand(5)
                    constr_type = "all_colors_all_values"

        elif joint_constr == "one_card":
            color_id = np.random.randint(0, 5)
            value_id = np.random.randint(0, 5)
            prob = np.random.rand()
            joint_probs[color_id, value_id] = prob
        else:
            joint_probs = softmax(np.random.rand(5, 5))

        constr_reversed = np.random.rand() > 0.95

        return StochasticCardPattern(color_id, value_id, joint_constr, constr_type, color_probs, value_probs, joint_probs,
                                     constr_reversed)


# Rules
class Rule:
    def __init__(self, action=None, msg=""):
        self.action = action                            # The action to perform

        self.last_activations = 0                       # Used for decaying inactive rules
        self.activations = 0                            # Current episode activations
        self.chosen = 0                                 # Current episode times it has been chosen to perform the action

        # Cumulative metrics (they are persistent)
        self.total_activations = 0                      # Total activations of this rule
        self.total_chosen = 0                           # Total number of times it has been chosen to perform the action
        self.reward = 0                                 # Cumulative reward assigned to this Rule
        self.fitness_history = []                       # Allows to calculate a mean and a variance
        self.last_fitness = 0.0                         # Last calculated fitness (the current fitness of the rule)

        self.triggers = []
        self.msg = msg

    def fitness(self, alpha):
        if self.reward == 0:
            return 0.0
        self.last_fitness = alpha * self.reward + (1 - alpha) * self.last_fitness
        self.fitness_history.append(self.last_fitness)
        return self.last_fitness

    def fitness_variance(self):
        v = np.array(self.fitness_history)
        return np.var(v)

    def utility(self):
        return self.last_fitness

    def reset_stats(self):
        self.total_activations = 0
        self.total_chosen = 0
        self.reward = 0
        self.fitness_history = []
        self.last_fitness = 0.0

    def toString(self):
        init_str = f"\tRule: {self.msg}" if isinstance(self, BasicRule) else f"Compound rule"
        action_str = f"\nAction: {self.action.move_type}." if self.action is not None else ""
        return f"{init_str}.\n\tCurrent episode statistics: Active: {self.activations}, chosen: {self.chosen}.\n\t" \
               f"Total statistics: Active: {self.total_activations}, " \
               f"chosen: {self.total_chosen}, reward: {self.reward:.3f}, fitness: {self.last_fitness}.{action_str}"

    def toCompleteString(self):
        s = self.toString()
        s += "\n"
        s += f"\tTriggers ({len(self.triggers)}):\n"
        for t in self.triggers:
            tr_str = t.toString() if type(t) is not TriggeredBasicRule else "Basic trigger"
            s += f"\t\t{tr_str}\n"
        s += "\n"
        return s


    @staticmethod
    def make_random(max_group_size, specific_type=None):
        if specific_type is not None:
            rule = specific_type.make_random()
        else:
            if np.random.rand() > 0.05:
                rule = CompoundRule.make_random(max_group_size)
            else:
                rule = BasicRule.make_random()

        action = Action.make_random()
        rule.action = action
        return rule

class CompoundRule(Rule):
    def __init__(self, rules, action=None):
        super().__init__(action=action)
        self.rules = rules                              # The nested rules (they need to be all active to trigger this)

    def add_rule(self, new_r):
        valid = True
        if type(new_r) is TokenRule:
            token_rules = [r for r in self.rules if type(r) is TokenRule]
            for tr in token_rules:
                if not new_r.is_compatible(tr):
                    valid = False
                    break
        elif type(new_r) is PlayersNumberRule:
            pn_rules = [r for r in self.rules if type(r) is PlayersNumberRule]
            for pn in pn_rules:
                if not new_r.is_compatible(pn):
                    valid = False
                    break
        if valid:
            self.rules.append(new_r)
            return True
        return False

    def is_similar(self, other):
        # It is similar if, for each rule, it does exist at least one similar rule in the other
        for r1 in self.rules:
            other_cand = [r for r in other.rules if type(r) is type(r1)]
            if len(other_cand) < 1:
                return False
            r_similar = True
            for r2 in other_cand:
                if not r1.is_similar(r2):
                    r_similar = False
            if not r_similar:
                return False
        return True

    @staticmethod
    def make_random(max_group_size):
        group_size = np.random.randint(2, max_group_size+1)

        rules = []
        for i in range(group_size):
            # Try till we generate a rule compatible with the others
            while 1:
                rule = BasicRule.make_random()

                valid = True
                if type(rule) is TokenRule:
                    token_rules = [r for r in rules if type(r) is TokenRule]
                    for tr in token_rules:
                        if not rule.is_compatible(tr):
                            valid = False
                            break
                elif type(rule) is PlayersNumberRule:
                    pn_rules = [r for r in rules if type(r) is PlayersNumberRule]
                    for pn in pn_rules:
                        if not rule.is_compatible(pn):
                            valid = False
                            break
                if valid:
                    rules.append(rule)
                    break

        return CompoundRule(rules, action=None)

    def toString(self):
        s = super().toString()
        s += "\n"
        s += f"Nested rules ({len(self.rules)}):\n"
        for r in self.rules:
            s += f"{r.toString()}"
            s += "\n\n"
        return s

    def toCompleteString(self):
        s = super().toString()
        s += "\n"
        s += f"Nested rules ({len(self.rules)}):\n"
        for r in self.rules:
            s += f"{r.toCompleteString()}"
            s += "\n\n"
        return s

class BasicRule(Rule):
    def __init__(self, action=None):
        super().__init__(action=action)

    @staticmethod
    def make_random():
        sources = [CardRule, FellowCardRule, StochasticCardRule, TokenRule, PlayersNumberRule]
        p = [15/100, 40/100, 40/100, 2.5/100, 2.5/100]
        r_type = np.random.choice(sources, p=p)

        return r_type.make_random()

class CardRule(BasicRule):
    owner_types = ["discard_pile", "table"]
    owner_types_probs = [0.5, 0.5]
    p_num_triggers = [0.1, 0.5, 0.1, 0.1, 0.1, 0.1]
    def __init__(self, owner_type, card_pattern: CardPattern, num_requested_trig, rev_num_req_trig, action=None):
        super().__init__(action=action)
        self.owner_type = owner_type
        self.card_pattern = card_pattern
        self.num_requested_trig = num_requested_trig
        self.rev_num_req_trig = rev_num_req_trig

        # Self-adaptation parameters
        self.num_req_trig_mutation = DiscreteMutation(initial_prob=0.01, learning_rate=0.1)

    def get_strength(self, candidate_card):
        if self.card_pattern.match(candidate_card):
            return 1.0
        else:
            return 0

    def is_similar(self, other):
        return self.owner_type == other.owner_type and self.card_pattern.is_similar(other.card_pattern) \
                and self.rev_num_req_trig == other.rev_num_req_trig

    def creep_mutation(self):
        # Creep mutation card pattern
        card_pattern = self.card_pattern
        constr_type = self.card_pattern.constr_type

        card_pattern.creep_mutation()

        # Creep mutation number of requested triggers
        rev_trig_constr = self.rev_num_req_trig
        trig_mutation = self.num_req_trig_mutation

        trig_mutation.mutate()

        min_constr = 2 if rev_trig_constr else 1
        if constr_type == "joint":
            max_constr = card_values_to_counts[card_pattern.value]
            if min_constr < max_constr:
                possible_values = list(range(min_constr, max_constr+1))
                self.num_requested_trig = trig_mutation.switch_value(self.num_requested_trig, possible_values)
        elif constr_type == "color_only" or constr_type == "value_only":
            possible_values = list(range(min_constr, 6))
            self.num_requested_trig = trig_mutation.switch_value(self.num_requested_trig, possible_values)

        self.update_msg()

    def crossover(self, other):
        if np.random.rand() > 0.5:
            return self.average_crossover(other)
        return self.uniform_crossover(other)

    def average_crossover(self, other):
        xover_card_pattern = self.card_pattern.average_crossover(other.card_pattern)
        constr_type = xover_card_pattern.constr_type

        rev_trig_constr = self.rev_num_req_trig
        xover_num_trig = np.abs(self.num_requested_trig - other.num_requested_trig) // 2

        min_constr = 2 if rev_trig_constr else 1
        if constr_type == "joint":
            max_constr = card_values_to_counts[xover_card_pattern.value]
            if min_constr < max_constr:
                xover_num_trig = np.clip(xover_num_trig, min_constr, max_constr)
            else:
                xover_num_trig = min_constr
        elif constr_type == "color_only" or constr_type == "value_only":
            xover_num_trig = np.clip(xover_num_trig, min_constr, 5)

        xover_rule = CardRule(self.owner_type, xover_card_pattern, xover_num_trig, rev_trig_constr, action=self.action)
        xover_rule.update_msg()
        return xover_rule

    def uniform_crossover(self, other):
        if np.random.rand() > 0.5:
            start = self
            end = other
        else:
            start = other
            end = self

        xover_card_pattern = start.card_pattern.uniform_crossover(end.card_pattern)

        constr_type = xover_card_pattern.constr_type
        rev_trig_constr = self.rev_num_req_trig

        xover_num_trig = start.num_requested_trig

        min_constr = 1 if rev_trig_constr else 0
        if constr_type == "joint":
            max_constr = card_values_to_counts[xover_card_pattern.value]
            if min_constr < max_constr:
                xover_num_trig = np.clip(xover_num_trig, min_constr, max_constr)
            else:
                xover_num_trig = min_constr
        elif constr_type == "color_only" or constr_type == "value_only":
            xover_num_trig = np.clip(xover_num_trig, min_constr, 5)

        xover_rule = CardRule(self.owner_type, xover_card_pattern, xover_num_trig, rev_trig_constr, action=self.action)
        xover_rule.update_msg()
        return xover_rule

    def update_msg(self):
        rev_str = "greater or equal than" if not self.rev_num_req_trig else "lower than"
        color = self.card_pattern.color
        value = self.card_pattern.value
        constr_type = self.card_pattern.constr_type
        if constr_type == "joint":
            if not self.card_pattern.constr_reversed:
                self.msg = f"The {self.owner_type} must have a number of ({value}-{color}) cards {rev_str} {self.num_requested_trig}."
            else:
                self.msg = f"The {self.owner_type} must have a number of cards different than ({value}-{color}) {rev_str} {self.num_requested_trig}."
        elif constr_type == "color_only":
            if not self.card_pattern.constr_reversed:
                self.msg = f"The {self.owner_type} must have a number of {color} cards {rev_str} {self.num_requested_trig}."
            else:
                self.msg = f"The {self.owner_type} must have a number of cards with a color different than {color} {rev_str} {self.num_requested_trig}."
        elif constr_type == "value_only":
            if not self.card_pattern.constr_reversed:
                self.msg = f"The {self.owner_type} must have a number of {value}-cards {rev_str} {self.num_requested_trig}."
            else:
                self.msg = f"The {self.owner_type} must have a number of cards with a value different than {value} {rev_str} {self.num_requested_trig}."
        else:
            raise Exception()

    @staticmethod
    def make_random(owner_type=None):
        # 1. Generate the owner
        if owner_type is None:
            owner_type = np.random.choice(CardRule.owner_types, p=CardRule.owner_types_probs)

        # 2. Generate the card pattern
        card_pattern = CardPattern.make_random()

        # 3. Number of requested triggers
        rev_num_req_trig = np.random.rand() > 0.9

        min_constr = 2 if rev_num_req_trig else 1
        if card_pattern.constr_type == "joint":
            max_constr = card_values_to_counts[card_pattern.value]
            if min_constr < max_constr:
                p = softmax(CardRule.p_num_triggers[min_constr:max_constr+1])
                num_trig = np.random.choice(list(range(min_constr, max_constr+1)), p=p)
                #num_trig = np.random.randint(min_constr, max_constr + 1)
            else:
                num_trig = min_constr
        elif card_pattern.constr_type == "color_only" or card_pattern.constr_type == "value_only":
            p = softmax(CardRule.p_num_triggers[min_constr:6])
            num_trig = np.random.choice(list(range(min_constr, 6)), p=p)

        cr = CardRule(owner_type, card_pattern, num_trig, rev_num_req_trig, action=None)
        cr.update_msg()

        return cr

class FellowCardRule(CardRule):
    def __init__(self, fellow_num, card_pattern: CardPattern, num_requested_trig, rev_num_req_trig, action=None):
        super().__init__(owner_type="fellow", card_pattern=card_pattern, num_requested_trig=num_requested_trig,
                         rev_num_req_trig=rev_num_req_trig, action=action)
        self.fellow_num = fellow_num

    def is_similar(self, other):
        if self.fellow_num is not None and other.fellow_num is None:
            return False
        if other.fellow_num is not None and self.fellow_num is None:
            return False
        return super().is_similar(other)

    def creep_mutation(self):
        super().creep_mutation()
        self.update_msg()

    def crossover(self, other):
        if np.random.rand() > 0.5:
            return self.average_crossover(other)
        return self.uniform_crossover(other)

    def average_crossover(self, other):
        xover_card_rule = super().average_crossover(other)

        xover_fellow_num = np.clip(np.abs(self.fellow_num - other.fellow_num) // 2, 1, 4) if self.fellow_num is not None else None

        xover_rule = FellowCardRule(xover_fellow_num, xover_card_rule.card_pattern, xover_card_rule.num_requested_trig,
                              xover_card_rule.rev_num_req_trig, xover_card_rule.action)
        xover_rule.update_msg()
        return xover_rule

    def uniform_crossover(self, other):
        xover_card_rule = super().uniform_crossover(other)

        xover_fellow_num = self.fellow_num if np.random.rand() > 0.5 else other.fellow_num

        xover_rule = FellowCardRule(xover_fellow_num, xover_card_rule.card_pattern, xover_card_rule.num_requested_trig,
                              xover_card_rule.rev_num_req_trig, xover_card_rule.action)
        xover_rule.update_msg()
        return xover_rule

    def update_msg(self):
        rev_str = "greater or equal than" if not self.rev_num_req_trig else "lower than"
        color = self.card_pattern.color
        value = self.card_pattern.value
        constr_type = self.card_pattern.constr_type
        fellow_str = f" which plays {self.fellow_num} turns after me " if self.fellow_num is not None else " "

        if constr_type == "joint":
            if not self.card_pattern.constr_reversed:
                self.msg = f"The fellow player{fellow_str}must have a number of ({value}-{color}) cards {rev_str} {self.num_requested_trig}."
            else:
                self.msg = f"The fellow player{fellow_str}must have a number of cards different than ({value}-{color}) {rev_str} {self.num_requested_trig}."
        elif constr_type == "color_only":
            if not self.card_pattern.constr_reversed:
                self.msg = f"The fellow player{fellow_str}must have a number of {color} cards {rev_str} {self.num_requested_trig}."
            else:
                self.msg = f"The fellow player{fellow_str}must have a number of cards with a color different than {color} {rev_str} {self.num_requested_trig}."
        elif constr_type == "value_only":
            if not self.card_pattern.constr_reversed:
                self.msg = f"The fellow player{fellow_str}must have a number of {value}-cards {rev_str} {self.num_requested_trig}."
            else:
                self.msg = f"The fellow player{fellow_str}must have a number of cards with a value different than {value} {rev_str} {self.num_requested_trig}."
        else:
            raise Exception()

    @staticmethod
    def make_random():
        cr = CardRule.make_random(owner_type="fellow")
        if np.random.rand() > 0.05:
            fellow_num = None
        else:
            fellow_num = np.random.randint(1, 5)

        fcr = FellowCardRule(fellow_num, cr.card_pattern, cr.num_requested_trig, cr.rev_num_req_trig, action=None)
        fcr.update_msg()
        return fcr

class StochasticCardRule(BasicRule):
    owner_types = ["myself", "deck"]
    owner_types_probs = [0.75, 0.25]
    p_num_triggers = [0.1, 0.5, 0.1, 0.1, 0.1, 0.1]
    def __init__(self, owner_type, stoc_card_pattern: StochasticCardPattern, num_requested_trig, rev_num_req_trig, action=None):
        super().__init__(action=action)
        self.owner_type = owner_type
        self.stoc_card_pattern = stoc_card_pattern
        self.num_requested_trig = num_requested_trig
        self.rev_num_req_trig = rev_num_req_trig

        # Self-adaptation parameters
        self.num_req_trig_mutation = DiscreteMutation(initial_prob=0.01, learning_rate=0.1)

    def get_strength(self, stoc_card: StochasticCard):
        return self.stoc_card_pattern.get_strength(stoc_card)

    def is_similar(self, other):
        return self.owner_type == other.owner_type and self.stoc_card_pattern.is_similar(other.stoc_card_pattern) \
                and self.rev_num_req_trig == other.rev_num_req_trig

    def creep_mutation(self):
        # Creep mutation card pattern
        stoc_card_pattern = self.stoc_card_pattern
        joint_constr = stoc_card_pattern.joint_constr

        stoc_card_pattern.creep_mutation()

        # Creep mutation of number of requested triggers
        rev_num_trig = self.rev_num_req_trig
        num_trig_mutation = self.num_req_trig_mutation
        min_constr = 2 if rev_num_trig else 1

        num_trig_mutation.mutate()

        if joint_constr == "one_card":
            max_constr = card_values_to_counts[stoc_card_pattern.value_id + 1]
            if min_constr < max_constr:
                possible_values = list(range(min_constr, max_constr + 1))
                self.num_requested_trig = num_trig_mutation.switch_value(self.num_requested_trig, possible_values)
        else:
            possible_values = list(range(min_constr, 6))
            self.num_requested_trig = num_trig_mutation.switch_value(self.num_requested_trig, possible_values)

    def crossover(self, other):
        if np.random.rand() > 0.5:
            return self.average_crossover(other)
        return self.uniform_crossover(other)

    def average_crossover(self, other):
        xover_stoc_card_pattern = self.stoc_card_pattern.average_crossover(other.stoc_card_pattern)
        joint_constr = xover_stoc_card_pattern.joint_constr

        rev_num_trig = self.rev_num_req_trig
        min_constr = 1 if rev_num_trig else 0

        xover_num_trig = np.abs(self.num_requested_trig - other.num_requested_trig) // 2

        if joint_constr == "one_card":
            max_constr = card_values_to_counts[xover_stoc_card_pattern.value_id + 1]
            if min_constr < max_constr:
                xover_num_trig = np.clip(xover_num_trig, min_constr, max_constr)
            else:
                xover_num_trig = min_constr
        else:
            xover_num_trig = np.clip(xover_num_trig, min_constr, 5)

        return StochasticCardRule(self.owner_type, xover_stoc_card_pattern, xover_num_trig, rev_num_trig, action=self.action)

    def uniform_crossover(self, other):
        if np.random.rand() > 0.5:
            start = self
            end = other
        else:
            start = other
            end = self

        xover_stoc_card_pattern = self.stoc_card_pattern.uniform_crossover(other.stoc_card_pattern)
        joint_constr = xover_stoc_card_pattern.joint_constr

        rev_num_trig = self.rev_num_req_trig
        min_constr = 2 if rev_num_trig else 1

        xover_num_trig = start.num_requested_trig
        if joint_constr == "one_card":
            max_constr = card_values_to_counts[xover_stoc_card_pattern.value_id + 1]
            if min_constr < max_constr:
                xover_num_trig = np.clip(xover_num_trig, min_constr, max_constr)
            else:
                xover_num_trig = min_constr
        else:
            xover_num_trig = np.clip(xover_num_trig, min_constr, 5)

        return StochasticCardRule(self.owner_type, xover_stoc_card_pattern, xover_num_trig, rev_num_trig, action=self.action)

    def update_msg(self):
        rev_str = "greater or equal than" if not self.stoc_card_pattern.constr_reversed else "lower than"
        num_trig_str = "greater or equal than" if not self.rev_num_req_trig else "lower than"
        num_trig = self.num_requested_trig
        joint_constr = self.stoc_card_pattern.joint_constr
        constr_type = self.stoc_card_pattern.constr_type
        color_id = self.stoc_card_pattern.color_id if self.stoc_card_pattern.color_id is not None else None
        color = StochasticCard.index_to_colors[color_id] if color_id is not None else None
        value = self.stoc_card_pattern.value_id + 1 if self.stoc_card_pattern.value_id is not None else None
        if joint_constr is None:
            if constr_type == "one_col":
                if self.owner_type == "myself":
                    self.msg = f"I must have a probability of having a number {num_trig_str} {num_trig} of cards with " \
                               f"color {color} {rev_str} {self.stoc_card_pattern.color_probs[color_id]}."
                else:
                    self.msg = f"We must have a probability of drawing a number {num_trig_str} {num_trig} of cards with " \
                               f"color {color} {rev_str} {self.stoc_card_pattern.color_probs[color_id]}."
            elif constr_type == "one_val":
                if self.owner_type == "myself":
                    self.msg = f"I must have a probability of having a number {num_trig_str} {num_trig} of cards with " \
                               f"value {value} {rev_str} {self.stoc_card_pattern.value_probs[value-1]}."
                else:
                    self.msg = f"We must have a probability of drawing a number {num_trig_str} {num_trig} of cards with " \
                               f"value {value} {rev_str} {self.stoc_card_pattern.value_probs[value-1]}."
            elif constr_type == "one_col_one_val":
                if self.owner_type == "myself":
                    self.msg = f"I must have a probability of having a number {num_trig_str} {num_trig} of cards with " \
                               f"value {value} or color {color} {rev_str} {self.stoc_card_pattern.value_probs[value - 1]}."
                else:
                    self.msg = f"We must have a probability of drawing a number {num_trig_str} {num_trig} of cards with " \
                               f"value {value} or color {color} {rev_str} {self.stoc_card_pattern.value_probs[value - 1]}."
            elif constr_type == "all_colors":
                if self.owner_type == "myself":
                    self.msg = f"I must have a probability of having a number {num_trig_str} {num_trig} of cards with " \
                               f"color probabilities {rev_str} {self.stoc_card_pattern.color_probs}."
                else:
                    self.msg = f"We must have a probability of drawing a number {num_trig_str} {num_trig} of cards with " \
                               f"color probabilities {rev_str} {self.stoc_card_pattern.color_probs}."
            elif constr_type == "all_values":
                if self.owner_type == "myself":
                    self.msg = f"I must have a probability of having a number {num_trig_str} {num_trig} of cards with " \
                               f"value probabilities {rev_str} {self.stoc_card_pattern.value_probs}."
                else:
                    self.msg = f"We must have a probability of drawing a number {num_trig_str} {num_trig} of cards with " \
                               f"value probabilities {rev_str} {self.stoc_card_pattern.value_probs}."
            elif constr_type == "all_colors_all_values":
                if self.owner_type == "myself":
                    self.msg = f"I must have a probability of having a number {num_trig_str} {num_trig} of cards with " \
                               f"value probabilities {rev_str} {self.stoc_card_pattern.value_probs} and color probabilities " \
                               f" {rev_str} {self.stoc_card_pattern.color_probs}."
                else:
                    self.msg = f"We must have a probability of drawing a number {num_trig_str} {num_trig} of cards with " \
                               f"value probabilities {rev_str} {self.stoc_card_pattern.value_probs} and color probabilities " \
                               f" {rev_str} {self.stoc_card_pattern.color_probs}."
            else:
                raise Exception()
        elif joint_constr == "one_card":
            if self.owner_type == "myself":
                self.msg = f"I must have a probability of having a number {num_trig_str} {num_trig} of ({value}-{color}) cards with " \
                           f"probability {rev_str} {self.stoc_card_pattern.joint_probs[color_id, value-1]}."
            else:
                self.msg = f"We must have a probability of drawing a number {num_trig_str} {num_trig} of ({value}-{color}) cards with " \
                           f"probability {rev_str} {self.stoc_card_pattern.joint_probs[color_id, value - 1]}."
        elif joint_constr == "all_cards":
            if self.owner_type == "myself":
                self.msg = f"I must have a probability of having a number {num_trig_str} {num_trig} of cards with " \
                           f"probabilities {rev_str} {self.stoc_card_pattern.joint_probs}."
            else:
                self.msg = f"We must have a probability of drawing a number {num_trig_str} {num_trig} of cards with " \
                           f"probabilities {rev_str} {self.stoc_card_pattern.joint_probs}."

    @staticmethod
    def make_random(owner_type=None):
        if owner_type is None:
            owner_type = np.random.choice(StochasticCardRule.owner_types, p=StochasticCardRule.owner_types_probs)

        # 2. Generate the card pattern
        stoc_card_pattern = StochasticCardPattern.make_random()

        # 3. Number of requested triggers
        rev_num_req_trig = np.random.rand() > 0.9
        min_constr = 2 if rev_num_req_trig else 1

        joint_constr = stoc_card_pattern.joint_constr
        if joint_constr == "one_card":
            max_constr = card_values_to_counts[stoc_card_pattern.value_id + 1]
            if min_constr < max_constr:
                p = softmax(StochasticCardRule.p_num_triggers[min_constr:max_constr+1])
                num_trig = np.random.choice(list(range(min_constr, max_constr+1)), p=p)
                # num_trig = np.random.randint(min_constr, max_constr + 1)
            else:
                num_trig = min_constr
        else:
            p = softmax(StochasticCardRule.p_num_triggers[min_constr:6])
            num_trig = np.random.choice(list(range(min_constr, 6)), p=p)
            #num_trig = np.random.randint(min_constr, 6)

        scr = StochasticCardRule(owner_type, stoc_card_pattern, num_trig, rev_num_req_trig, action=None)
        scr.update_msg()
        return scr

class TokenRule(BasicRule):
    token_types = ["note", "storm"]
    def __init__(self, token_type, constraint, constr_reversed, action=None):
        super().__init__(action=action)
        self.token_type = token_type
        self.constraint = constraint
        self.constr_reversed = constr_reversed

        # Self-adaptation parameters
        self.constraint_mutation = DiscreteMutation(initial_prob=0.6, learning_rate=1)

    def get_strength(self, note_tokens, storm_tokens):
        if self.token_type == "note":
            tk = note_tokens
            maximum = 8
        elif self.token_type == "storm":
            tk = storm_tokens
            maximum = 3
        else:
            raise Exception()

        if not self.constr_reversed:
            if tk >= self.constraint:
                return (tk - self.constraint + 1) / maximum
            else:
                return 0
        else:
            if tk < self.constraint:
                return (self.constraint - tk) / maximum
            else:
                return 0

    def is_similar(self, other):
        return self.token_type == other.token_type and self.constr_reversed == other.constr_reversed

    def is_compatible(self, other):
        if self.token_type == other.token_type:
            if self.constr_reversed == other.constr_reversed:
                return False
            if not self.constr_reversed:
                if self.constraint >= other.constraint:
                    return False
            else:
                if self.constraint <= other.constraint:
                    return False
        return True

    def creep_mutation(self):
        rev = self.constr_reversed
        max_note_constr = 8 if not rev else 9
        min_note_constr = 0 if not rev else 1
        max_storm_constr = 2 if not rev else 3
        min_storm_constr = 0 if not rev else 1
        values = list(range(min_note_constr, max_note_constr+1)) if self.token_type == "note" else list(range(min_storm_constr, max_storm_constr+1))
        self.constraint_mutation.mutate()
        self.constraint = self.constraint_mutation.switch_value(self.constraint, values)
        self.update_msg()

    def crossover(self, other):
        if np.random.rand() > 0.5:
            return self.average_crossover(other)
        return self.uniform_crossover(other)

    def average_crossover(self, other):
        xover_constraint = np.abs(self.constraint - other.constraint) // 2
        rev = self.constr_reversed
        max_note_constr = 8 if not rev else 9
        min_note_constr = 0 if not rev else 1
        max_storm_constr = 2 if not rev else 3
        min_storm_constr = 0 if not rev else 1
        min_constr = min_note_constr if self.token_type == "note" else min_storm_constr
        max_constr = max_note_constr if self.token_type == "note" else max_storm_constr

        xover_constraint = np.clip(xover_constraint, min_constr, max_constr)
        xover_rule = TokenRule(self.token_type, xover_constraint, rev, action=self.action)
        xover_rule.update_msg()
        return xover_rule

    def uniform_crossover(self, other):
        if np.random.rand() > 0.5:
            start = self
            end = other
        else:
            start = other
            end = self

        xover_constraint = start.constraint
        rev = start.constr_reversed
        max_note_constr = 8 if not rev else 9
        min_note_constr = 0 if not rev else 1
        max_storm_constr = 2 if not rev else 3
        min_storm_constr = 0 if not rev else 1
        min_constr = min_note_constr if self.token_type == "note" else min_storm_constr
        max_constr = max_note_constr if self.token_type == "note" else max_storm_constr

        xover_constraint = np.clip(xover_constraint, min_constr, max_constr)
        xover_rule = TokenRule(self.token_type, xover_constraint, rev, action=start.action)
        xover_rule.update_msg()
        return xover_rule

    def update_msg(self):
        rev_str = "greater or equal than" if not self.constr_reversed else "lower than"
        self.msg = f"Token rule - The used {self.token_type} tokens must be {rev_str} {self.constraint}."

    @staticmethod
    def make_random():
        rev = np.random.rand() > 0.5

        max_note_constr = 8 if not rev else 9
        min_note_constr = 0 if not rev else 1
        max_storm_constr = 2 if not rev else 3
        min_storm_constr = 0 if not rev else 1

        t_type = np.random.choice(TokenRule.token_types)
        if t_type == "note":
            constraint = np.random.randint(min_note_constr, max_note_constr+1)
        else:
            constraint = np.random.randint(min_storm_constr, max_storm_constr+1)

        tk = TokenRule(t_type, constraint, rev, action=None)
        tk.update_msg()

        return tk

class PlayersNumberRule(BasicRule):
    def __init__(self, constraint, constr_reversed, action):
        super().__init__(action=action)
        self.constraint = constraint
        self.constr_reversed = constr_reversed

        # Self-adaptation parameters
        self.constraint_mutation = DiscreteMutation(initial_prob=0.6, learning_rate=1)

    def get_strenght(self, num_players):
        if not self.constr_reversed:
            if num_players >= self.constraint:
                return (num_players - self.constraint + 1) / num_players
            else:
                return 0
        else:
            if num_players < self.constraint:
                return (self.constraint - num_players) / num_players
            else:
                return 0

    def is_similar(self, other):
        return self.constr_reversed == other.constr_reversed

    def is_compatible(self, other):
        if self.constr_reversed == other.constr_reversed:
            return False
        if not self.constr_reversed:
            if self.constraint >= other.constraint:
                return False
        else:
            if self.constraint <= other.constraint:
                return False
        return True

    def creep_mutation(self):
        rev = self.constr_reversed
        min_constr = 2 if not rev else 3
        max_constr = 5 if not rev else 6

        self.constraint_mutation.mutate()
        possible_values = list(range(min_constr, max_constr+1))
        self.constraint = self.constraint_mutation.switch_value(self.constraint, possible_values)
        self.update_msg()

    def crossover(self, other):
        if np.random.rand() > 0.5:
            return self.average_crossover(other)
        return self.uniform_crossover(other)

    def average_crossover(self, other):
        rev = self.constr_reversed
        min_constr = 2 if not rev else 3
        max_constr = 5 if not rev else 6

        xover_constraint = np.abs(self.constraint - other.constraint) // 2
        xover_constraint = np.clip(xover_constraint, min_constr, max_constr)
        xover_rule = PlayersNumberRule(xover_constraint, rev, action=self.action)
        xover_rule.update_msg()
        return xover_rule

    def uniform_crossover(self, other):
        if np.random.rand() > 0.5:
            start = self
            end = other
        else:
            start = other
            end = self

        rev = self.constr_reversed
        min_constr = 2 if not rev else 3
        max_constr = 5 if not rev else 6

        xover_constraint = start.constraint
        xover_constraint = np.clip(xover_constraint, min_constr, max_constr)
        xover_rule = PlayersNumberRule(xover_constraint, rev, action=start.action)
        xover_rule.update_msg()
        return xover_rule

    def update_msg(self):
        rev_str = "greater or equal than" if not self.constr_reversed else "lower than"
        self.msg = f"Players number rule - The number of players must be {rev_str} {self.constraint}."

    @staticmethod
    def make_random():
        rev = np.random.rand() > 0.5
        min_constr = 2 if not rev else 3
        max_constr = 5 if not rev else 6
        constr = np.random.randint(min_constr, max_constr+1)

        pnr = PlayersNumberRule(constr, rev, action=None)
        pnr.update_msg()
        return pnr


# Actions
move_types = ["play", "discard", "hint_color", "hint_value"]
class Action:
    def __init__(self, rule, move_type):
        self.rule = rule                        # This represents the pattern + constraints to target something specific for the action
        self.move_type = move_type

        # Self-adaptation parameters
        self.move_type_mutation = DiscreteMutation(initial_prob=0.1, learning_rate=0.1)

    def creep_mutation(self):
        self.rule.creep_mutation()

        move_mutation = self.move_type_mutation
        move_mutation.mutate()

        if type(self.rule) is StochasticCardRule:
            possible_values = ["play", "discard"]
        elif type(self.rule) is FellowCardRule:
            possible_values = ["hint_color", "hint_value"]
        else:
            raise Exception()

        self.move_type = move_mutation.switch_value(self.move_type, possible_values)

    @staticmethod
    def make_random():
        move = np.random.choice(move_types)

        if move == "play" or move == "discard":
            target_rule = StochasticCardRule.make_random(owner_type="myself")
        else:
            target_rule = FellowCardRule.make_random()

        return Action(target_rule, move)


# TriggeredRules
class TriggeredRule:
    def __init__(self, rule):
        self.rule = rule

class TriggeredCompoundRule(TriggeredRule):
    def __init__(self, rule: CompoundRule):
        super().__init__(rule)

class TriggeredBasicRule(TriggeredRule):
    def __init__(self, rule, strength):
        super().__init__(rule)
        self.strength = strength

class TriggeredCardRule(TriggeredBasicRule):
    def __init__(self, rule: CardRule, strength, hint_card: HintCard):
        super().__init__(rule, strength)
        self.hint_card = hint_card

    def toString(self):
        hint_card = self.hint_card
        return f"Trigger: {hint_card.toString()} - strength: {self.strength:.3f}."

class TriggeredFellowCardRule(TriggeredCardRule):
    def __init__(self, rule: CardRule, strength, hint_card: HintCard, fellow):
        super().__init__(rule, strength, hint_card)
        self.fellow = fellow

    def toString(self):
        hint_card = self.hint_card
        fellow = self.fellow
        return f"Trigger: Fellow name: {fellow.name} - {hint_card.toString()} - strength: {self.strength:.3f}."

class TriggeredStochasticCardRule(TriggeredBasicRule):
    def __init__(self, rule: StochasticCardRule, strength, stoc_card: StochasticCard):
        super().__init__(rule, strength)
        self.stoc_card = stoc_card

    def toString(self):
        sc_str = self.stoc_card.card.toString() if self.stoc_card.card is not None else "Probability match"
        return f"Trigger: {sc_str} - strength: {self.strength:.3f}."

# Recognizing hints
class HintGiven:
    def __init__(self, fellow, hint_card):
        self.fellow = fellow
        self.hint_card = hint_card
        self.color_rules = []
        self.value_rules = []


# -------- GENETIC OPERATORS ---------

# Parent selection
def tournment_selection(rules, n, criterion):
    winners = []
    available = list(rules)
    for _ in range(n):
        i1 = np.random.randint(0, len(available))
        i2 = np.random.randint(0, len(available))
        r1 = available[i1]
        r2 = available[i2]
        winner = criterion(r1, r2)
        winners.append(winner)
        available = [r for i, r in enumerate(available) if i != i1 and i != i2]
    return winners

# Rules deletion (survival selection)
def __rule_deletion(r, log):
    activations = np.array([nr.total_activations for nr in r.rules])
    mean = np.mean(activations)
    if math.isclose(mean, 0.0):
        mean = 1.0  # Ensure at the start of the training that every rule is deleted and recreated till it gets active

    new_rules = []
    deleted_rules = []
    for nr in r.rules:
        if nr.total_activations < mean:
            if type(nr) is CompoundRule:
                __rule_deletion(nr)
                new_rules.append(nr)
            else:
                deleted_rules.append(nr)
        else:
            new_rules.append(nr)

    for dr in deleted_rules:
        if type(dr) is CompoundRule:
            raise Exception()

        new_rule = Rule.make_random(0, specific_type=BasicRule)
        new_rules.append(new_rule)
    r.rules = new_rules

def rule_deletion(rules, log):
    """
    Our goal here is to do survival selection, we discard inviduals with under average activations and under average
    utility (i.e. dead or weak rules)
    :param rules:
    :param log:
    :return:
    """
    activations = np.array([r.total_activations for r in rules])
    utilities = np.array([r.utility() for r in rules])
    mean_act = np.mean(activations)
    mean_ut = np.mean(utilities)
    if math.isclose(mean_act, 0.0):
        mean_act = 1.0  # Ensure at the start of the training that every rule is deleted and recreated till it gets active
    if math.isclose(mean_ut, 0.0) or mean_ut < 0.0:
        mean_ut = 1.0

    new_rules = []
    deleted_rules = []
    for r in rules:
        if r.total_activations < mean_act and r.utility() < mean_ut:
            if type(r) is CompoundRule:
                __rule_deletion(r, log)
                new_rules.append(r)
            else:
                deleted_rules.append(r)
        else:
            new_rules.append(r)

    log.info(f"Rule deletion: deleted and regenerated {len(deleted_rules)} / {len(rules)} rules.")

    for dr in deleted_rules:
        if type(dr) is CompoundRule:
            raise Exception()

        new_rule = Rule.make_random(0, specific_type=dr.__class__)
        new_rules.append(new_rule)

    return new_rules

# Rules specialization and generalization (based on Utility / Total activations)
def __comp_rule_specialization_generalization(cr: CompoundRule, quantity_factor=0.5, specialization=True):
    s_str = "specialization" if specialization else "generalization"
    # print(f"Nested rules before {s_str}: {len(cr.rules)}")
    #x = [(i, r) for i, r in enumerate(cr.rules)]

    # 1. Sort the nested rules by total_activations (in reverse order if we are specializing)
    #sort_rules = sorted(x, key=lambda r: r[1].total_activations, reverse=specialization)

    # 2. Extract some candidates
    def criterion(r1, r2):
        if specialization:
            if r1.total_activations > r2.total_activations:
                return r1
            else:
                return r2
        else:
            if r1.total_activations > r2.total_activations:
                return r2
            else:
                return r1


    l = int(len(cr.rules) * quantity_factor)
    candidates = tournment_selection(cr.rules, l, criterion)
    # candidates = sort_rules[:l]

    # 3. Creep mutation on them
    for c in candidates:
        if type(c) is CompoundRule:
            __comp_rule_specialization_generalization(c, quantity_factor=quantity_factor, specialization=specialization)
        else:
            c.creep_mutation()
            c.reset_stats()

    #if specialization:
        # print(f"Compound Rule specialization: creep mutation on {len(candidates)} / {len(cr.rules)} nested rules.")
    #else:
        # print(f"Compound Rule generalization: creep mutation on {len(candidates)} / {len(cr.rules)} nested rules.")

    if specialization:
        # Specialization: Adding a new rule
        while 1:
            new_r = BasicRule.make_random()
            if cr.add_rule(new_r):
                # print("Compound Rule specialization: added a new rule")
                break
    else:
        # Generalization: remove the rule with the highest total_activations
        # print("Compound Rule generalization: removed a rule")
        if len(cr.rules) > 2:
            cr.rules.pop()

    cr.reset_stats()
    # Modify the action (Compound rule -> action)
    cr.action.rule.creep_mutation()
    cr.action.creep_mutation()

    # print(f"Nested rules after {s_str}: {len(cr.rules)}")

def __rules_specialization_generalization(rules, quantity_factor=0.5, specialization=True):
    sort_rules = sorted(rules, key=lambda r: r.utility() / (r.total_activations + 1), reverse=(not specialization))

    def criterion(r1, r2):
        if specialization:
            if r1.utility() / (r1.total_activations + 1) > r2.utility() / (r2.total_activations + 1):
                return r2
            else:
                return r1
        else:
            if r1.utility() / (r1.total_activations + 1) > r2.utility() / (r2.total_activations + 1):
                return r1
            else:
                return r2

    l = int(len(rules) * quantity_factor)
    candidates = tournment_selection(rules, l, criterion)
    # candidates = sort_rules[:l]

    for c in candidates:
        if type(c) is CompoundRule:
            __comp_rule_specialization_generalization(c, quantity_factor=quantity_factor, specialization=specialization)
        else:
            c.creep_mutation()
            c.reset_stats()
            c.action.rule.creep_mutation()
            c.action.creep_mutation()

def rule_specialization(rules, log, quantity_factor=0.5):
    """
    We want to specialize the rules with low utility and high coverage (utility / fitness low).
    In order to do that, for the basic rules we simply mutate them, for the compounded rules, we repeat this process for
    its nested rules and we add an extra rule to specialize it (see __rule_specialization), we also mutate the connected
    action doing a creep mutation on the target rule and on the type of action to perform as well.
    :param rules:
    :param quantity_factor:
    :return:
    """
    __rules_specialization_generalization(rules, quantity_factor=quantity_factor, specialization=True)

def rule_generalization(rules, log, quantity_factor=0.5):
    """
    As the opposite, we want now to generalize the rules with high utility but low coverage (utility / fitness high).
    To do that, again, for the basic rules we mutate them, for the compounded rules, we repeat this process for its
    nested rules and we remove a rule to generalize it.
    :param rules:
    :param quantity_factor:
    :return:
    """
    __rules_specialization_generalization(rules, quantity_factor=quantity_factor, specialization=False)

# Rules merging - merging strong rules sharing the same action

def __merge_comp_rules(cr1: CompoundRule, cr2: CompoundRule):
    # For each rule in cr1, try to merge it with a similar rule in cr2
    min_len = min(len(cr1.rules), len(cr2.rules))
    max_len = max(len(cr1.rules), len(cr2.rules))
    new_rules = []
    index_r1_used, index_r2_used = [], []
    for i1, r1 in enumerate(cr1.rules):
        if i1 in index_r1_used or type(r1) is CompoundRule:
            continue
        valid_companions = [(i2, r2) for i2, r2 in enumerate(cr2.rules) if i2 not in index_r2_used \
                            and type(r1) is type(r2) and r1.is_similar(r2)]
        if len(valid_companions) > 0:
            i2, r2 = valid_companions[0]
            # Merge the rules
            merged_rule = r1.crossover(r2)

            new_rules.append(merged_rule)
            index_r1_used.append(i1)
            index_r2_used.append(i2)

    # Merge the target rules that individuate the target of the action
    merged_action_target_rule = cr1.action.rule.crossover(cr2.action.rule)
    new_comp_rule = CompoundRule(new_rules, action=cr1.action)
    new_comp_rule.action.rule = merged_action_target_rule

    # Maintain the same length as the longest compound rule
    if max_len == len(cr1.rules):
        others_2 = [r for i, r in enumerate(cr2.rules) if i not in index_r2_used]
        for r in others_2:
            new_comp_rule.add_rule(r)
    elif max_len == len(cr2.rules):
        others_1 = [r for i, r in enumerate(cr1.rules) if i not in index_r1_used]
        for r in others_1:
            new_comp_rule.add_rule(r)

    for _ in range(min_len - len(new_rules)):
        while 1:
            if new_comp_rule.add_rule(BasicRule.make_random()):
                break
    # print(f"Compound rule merge: minimum nested rules before merging: {min_len}, maximum: {max_len}, after merging: {len(new_comp_rule.rules)}")
    return new_comp_rule

def __merge_basic_rules(br1: BasicRule, br2: BasicRule):
    xover = br1.crossover(br2)
    merged_action_target_rule = br1.action.rule.crossover(br2.action.rule)
    xover.action = br1.action
    xover.action.rule = merged_action_target_rule
    return xover

def rule_merging(rules, max_group_size, log, quantity_factor=0.5):
    # 1. Sort the rules by utility

    sort_rules = sorted(rules, key=lambda r: r.utility(), reverse=True)

    # 2. Extract some candidates to try merge (selection)
    l = int(len(rules) * quantity_factor)
    candidates = sort_rules[:l]
    others = sort_rules[l:]

    cnt_basic_merged, cnt_comp_merged = 0, 0

    new_rules = []
    # 3. For each candidate, find the strongest similar candidate and merge them
    index_used = []
    for i, c in enumerate(candidates):
        if i in index_used:
            continue
        valid_companions = [(vc_i, vc) for vc_i, vc in enumerate(candidates) if vc_i != i and vc_i not in index_used \
                            and type(vc) is type(c) and vc.is_similar(c) and c.action.move_type == vc.action.move_type \
                            and c.action.rule.is_similar(vc.action.rule)]
        sorted_valid_companions = sorted(valid_companions, key=lambda vc: vc[1].utility(), reverse=True)
        if len(sorted_valid_companions) > 0:
            best_vc_i, best_vc = sorted_valid_companions[0]
            if type(best_vc) is CompoundRule:
                merged_rule = __merge_comp_rules(c, best_vc)
                #print("Merged comp rules")
                new_rules.append(merged_rule)
                cnt_comp_merged += 1
            else:
                merged_rule = __merge_basic_rules(c, best_vc)
                #print("Merged basic rules")
                new_rules.append(merged_rule)
                cnt_basic_merged += 1
            index_used.append(i)
            index_used.append(best_vc_i)


    log.info(f"Rule merging: merged {cnt_basic_merged} couples of basic rules and {cnt_comp_merged} couples of compound rules out of {len(rules)} original rules.")

    # 4. Remove the rules used for merging and adding the merged ones
    candidates.extend(new_rules)
    diff = len(others) - len(new_rules)
    candidates.extend(others[:diff])


    return candidates

# -------- REWARDS --------

REWARD_SCORE_LOSE = -70
REWARD_SCORE_MULTIPLIER_0_5 = 3
REWARD_SCORE_MULTIPLIER_5_10 = 4
REWARD_SCORE_MULTIPLIER_10_15 = 5
REWARD_SCORE_MULTIPLIER_15_20 = 7
REWARD_SCORE_MULTIPLIER_20_24 = 10
REWARD_SCORE_MULTIPLIER_25 = 15

REWARD_BAD_PLAY_1 = -3
REWARD_BAD_PLAY_234 = -4
REWARD_BAD_PLAY_5 = -5
REWARD_BAD_PLAY = [REWARD_BAD_PLAY_1, REWARD_BAD_PLAY_234, REWARD_BAD_PLAY_234, REWARD_BAD_PLAY_234, REWARD_BAD_PLAY_5]
REWARD_GOOD_PLAY_1 = 1
REWARD_GOOD_PLAY_2 = 2
REWARD_GOOD_PLAY_3 = 5
REWARD_GOOD_PLAY_4 = 7
REWARD_GOOD_PLAY_5 = 10
REWARD_GOOD_PLAY = [REWARD_GOOD_PLAY_1, REWARD_GOOD_PLAY_2, REWARD_GOOD_PLAY_3, REWARD_GOOD_PLAY_4, REWARD_GOOD_PLAY_5]

# Discard
# My discard
REWARD_STACK_NOT_BLOCKED_GOOD_DISCARD_UNUSEFUL_CARD = 1
REWARD_STACK_NOT_BLOCKED_DISCARD_REMAINING_2 = -1
REWARD_STACK_NOT_BLOCKED_DISCARD_REMAINING_1 = -2

REWARD_STACK_BLOCKED_BAD_DISCARD_BLOCK_STACK = -10
REWARD_STACK_BLOCKED_GOOD_DISCARD_ALREADY_BLOCKED = 2
REWARD_STACK_BLOCKED_ON_GREATER_CARD = 1

# Hint
# Fellow player plays with our hints
REWARD_GOOD_PLAY_AFTER_HINT = 2
REWARD_BAD_WRONG_PLAY_AFTER_HINT = -1
REWARD_DUPLICATED_HINT = -10

# Fellow player discards with our hints
REWARD_FELLOW_STACK_NOT_BLOCKED_GOOD_DISCARD_UNUSEFUL_CARD = 1
REWARD_FELLOW_STACK_NOT_BLOCKED_DISCARD_REMAINING_2 = 0
REWARD_FELLOW_STACK_NOT_BLOCKED_DISCARD_REMAINING_1 = 0

REWARD_FELLOW_STACK_BLOCKED_BAD_DISCARD_BLOCK_STACK = -2
REWARD_FELLOW_STACK_BLOCKED_GOOD_DISCARD_ALREADY_BLOCKED = 1
REWARD_FELLOW_STACK_BLOCKED_ON_GREATER_CARD = 0

def score_to_reward(score):
    if score == 0:
        reward = REWARD_SCORE_LOSE
    elif 0 < score <= 5:
        reward = score * REWARD_SCORE_MULTIPLIER_0_5
    elif 5 < score <= 10:
        reward = score * REWARD_SCORE_MULTIPLIER_5_10
    elif 10 < score <= 15:
        reward = score * REWARD_SCORE_MULTIPLIER_10_15
    elif 15 < score <= 20:
        reward = score * REWARD_SCORE_MULTIPLIER_15_20
    elif 20 < score <= 24:
        reward = score * REWARD_SCORE_MULTIPLIER_20_24
    elif score == 25:
        reward = score * REWARD_SCORE_MULTIPLIER_25
    else:
        raise Exception()

    return reward


# Mutation and its derivates offer a simple container for the mutation parameters (Self-adaptation)
# It offers a self-adaptation II <v1, ..., vn, sigma1, ... sigman>
class Mutation:
    def __init__(self):
        pass

class ContinuousMutation(Mutation):
    def __init__(self, initial_sigma, learning_rate):
        super().__init__()
        self.sigma = initial_sigma
        self.learning_rate = learning_rate
        self.n = 1

    def mutate(self):
        k = np.random.normal(0, 1)
        tao = self.learning_rate / np.sqrt(self.n)
        self.sigma = self.sigma * np.exp(tao * k)
        self.n += 0.01
        return self.sigma

class DiscreteMutation(Mutation):
    def __init__(self, initial_prob, learning_rate):
        super().__init__()
        self.prob = initial_prob
        self.learning_rate = learning_rate
        self.n = 1

    def mutate(self):
        k = np.random.normal(0, 1)
        tao = self.learning_rate / np.sqrt(self.n)
        self.prob = self.prob * np.exp(tao * k)
        self.prob = np.clip(self.prob, 0.0, 1.0)
        self.n += 0.01
        return self.prob

    def switch_value(self, curr_value, possible_values):
        possible_values = list(possible_values)
        if curr_value in possible_values:
            possible_values.remove(curr_value)
        if len(possible_values) > 1:
            k = np.random.rand()
            if k < self.prob:
                return np.random.choice(possible_values)
        elif len(possible_values) == 1:
            return possible_values[0]

        return curr_value
