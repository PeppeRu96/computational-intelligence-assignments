# Copyright © 2021 Giovanni Squillero <squillero@polito.it>
# Free for personal or classroom use; see 'LICENCE.md' for details.
# https://github.com/squillero/computational-intelligence

import logging
from math import sqrt
from typing import Any
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

NUM_CITIES = 23


class Tsp:

    def __init__(self, num_cities: int, seed: Any = None) -> None:
        if seed is None:
            seed = num_cities
        self._num_cities = num_cities
        self._graph = nx.DiGraph()
        self.eval_cnt = 0
        np.random.seed(seed)
        for c in range(num_cities):
            self._graph.add_node(c, pos=(np.random.random(), np.random.random()))

    def distance(self, n1, n2) -> int:
        pos1 = self._graph.nodes[n1]['pos']
        pos2 = self._graph.nodes[n2]['pos']
        return round(1_000_000 / self._num_cities * sqrt((pos1[0] - pos2[0])**2 +
                                                         (pos1[1] - pos2[1])**2))

    def evaluate_solution(self, solution: np.array) -> float:
        self.eval_cnt += 1
        total_cost = 0
        tmp = solution.tolist() + [solution[0]]
        for n1, n2 in (tmp[i:i + 2] for i in range(len(tmp) - 1)):
            total_cost += self.distance(n1, n2)
        return total_cost

    def plot(self, path: np.array = None, title = None) -> None:
        if path is not None:
            self._graph.remove_edges_from(list(self._graph.edges))
            tmp = path.tolist() + [path[0]]
            for n1, n2 in (tmp[i:i + 2] for i in range(len(tmp) - 1)):
                self._graph.add_edge(n1, n2)
        plt.figure(figsize=(12, 5))
        ax = plt.gca()
        if path is not None:
            ax.set_title(f"{title} - Current path: {self.evaluate_solution(path):,}")
        nx.draw(self._graph,
                pos=nx.get_node_attributes(self._graph, 'pos'),
                with_labels=True,
                node_color='pink',
                ax=ax)
        _ = ax.axis('off')

        #plt.show()
    @property
    def graph(self) -> nx.digraph:
        return self._graph

# Useful for hillclimbing
def tweak_randomly(solution: np.array, *, pm: float = .1) -> np.array:
    new_solution = solution.copy()
    p = None
    while p is None or p < pm:
        i1 = np.random.randint(0, solution.shape[0])
        i2 = np.random.randint(0, solution.shape[0])
        temp = new_solution[i1]
        new_solution[i1] = new_solution[i2]
        new_solution[i2] = temp
        p = np.random.random()
    return new_solution

def is_valid(genome, genome_length):
    for i in range(genome_length):
        if i not in genome:
            return False

    return True

def tournment_selection(population, eval_f, tournment_size=2, cost_minimization=False):
    cand_idx = np.random.randint(0, population.shape[0], size=(tournment_size,))
    candidates = population[cand_idx]
    evals = np.array([eval_f(candidates[i]) for i in range(tournment_size)])

    if cost_minimization:
        arg_f = np.argmin
    else:
        arg_f = np.argmax

    return candidates[arg_f(evals)]

def roulette_wheel_lin_selection(population, eval_f, cost_minimization=False):
    evals = np.array([eval_f(population[i]) for i in range(population.shape[0])])

    idx = evals.argsort()
    if cost_minimization is False:
        idx = idx[::-1]

    p = np.logspace(1, population.shape[0], population.shape[0], base=2)
    p = 1 / p
    i = np.random.choice(np.arange(0, population.shape[0]), p=p)
    return population[idx[i]]

def insert_mutation(p, genome_length):
    i1, i2 = 0, 0
    while i1 == i2 or abs(i1-i2) < 2:
        i1 = np.random.randint(0, genome_length)
        i2 = np.random.randint(0, genome_length)

    if i2 < i1:
        i1, i2 = i2, i1

    o = p.copy()
    o[i1+1] = p[i2]
    o[i1+2:i2+1] = p[i1+1:i2]

    #assert is_valid(o, genome_length)

    return o

def inversion_mutation(p, genome_length):
    i1, i2 = 0, 0
    while i1 == i2 or abs(i1 - i2) < 2:
        i1 = np.random.randint(0, genome_length)
        i2 = np.random.randint(0, genome_length)

    if i2 < i1:
        i1, i2 = i2, i1

    o = p.copy()
    if i1 == 0:
        o[i1 : i2+1] = p[i2 : : -1]
    else:
        o[i1 : i2+1] = p[i2 : i1-1 : -1]

    #assert is_valid(o, genome_length)

    return o

def _inver_over_xover(p1, p2, genome_length):
    # Generate a random index to start with one locus from p1
    i_p1 = np.random.randint(0, genome_length)
    v1 = p1[i_p1]
    # Get the corresponding locus from p2
    i_p2 = np.where(p2 == v1)[0][0]
    # Get the successive locus from p2
    next_i_p2 = (i_p2 + 1) % genome_length
    v2 = p2[next_i_p2]

    # Get the locus of p1 containing v2
    i2_p1 = np.where(p1 == v2)[0][0]
    # Get the sequence to be inherited from p1
    i_start, i_end = min(i_p1, i2_p1), max(i_p1, i2_p1)

    # Build an offspring
    o = p1.copy()
    # 1. Inherit v1->v2 link from p2
    o[i_start] = v1
    o[i_start + 1] = v2

    # 2. Inherit the inverted sequence from p1
    seq_len = i_end - i_start - 1
    i_o_start = i_start + 2
    o[i_o_start: i_o_start + seq_len] = p1[i_end - 1: i_start: -1]

    # At this point the offspring should be legal
    #print(f"\n\ti1:\t\t{i_p1}\n\ti_start:\t\t{i_start}\n\t{'p1:':7}\t\t{p1}\n\t{'p2:':7}\t\t{p2}\n\t{'Genome:':7}\t\t{o}")
    #assert is_valid(o, genome_length), f"\n\ti1:\t\t{i_p1}\n\ti_start:\t\t{i_start}\n\t{'p1:':7}\t\t{p1}\n\t{'p2:':7}\t\t{p2}\n\t{'Genome:':7}\t\t{o} is not valid"

    return o

def inver_over_xover(p1, p2, genome_length):
    o1 = _inver_over_xover(p1, p2, genome_length)
    o2 = _inver_over_xover(p2, p1, genome_length)

    return o1, o2

def parent_selection(problem, population):
    return tournment_selection(population, problem.evaluate_solution, tournment_size=2, cost_minimization=True)

def tweak_xover_plus_mutation(p1, p2, genome_length, pm=0.2, px=0.3):
    o1, o2 = p1.copy(), p2.copy()
    if np.random.random() < px:
        o1, o2 = inver_over_xover(p1, p2, genome_length)

    while np.random.random() < pm:
        if np.random.random() < 0.5:
            o1, o2 = insert_mutation(o1, genome_length), insert_mutation(o2, genome_length)
        else:
            o1, o2 = inversion_mutation(o1, genome_length), inversion_mutation(o2, genome_length)

    return o1, o2

def tsp_population_based_solver(problem, population_size, offspring_size, steady_state, pm=0.2, px=0.3):
    problem.eval_cnt = 0
    # Randomly initializing population
    population = np.array([np.random.permutation(NUM_CITIES) for _ in range(population_size)])

    # Steady-state termination
    curr_steady_count = 0
    generations = 1
    global_best = population[0]
    global_min_cost = problem.evaluate_solution(population[0])

    problem.plot(global_best, title="Population-based - Initial")
    print(f"Initial solution cost: {global_min_cost:,}")

    while curr_steady_count < steady_state:
        generations += 1
        curr_steady_count += 1
        # Generate offspring
        offspring = []
        while len(offspring) < offspring_size:
            # Parent selection
            p1, p2 = parent_selection(problem, population), parent_selection(problem, population)

            # Crossover plus mutation
            o1, o2 = tweak_xover_plus_mutation(p1, p2, NUM_CITIES, pm=pm, px=px)
            offspring.append(o1)
            offspring.append(o2)
        offspring = np.array(offspring)

        # Survival selection
        costs = np.array([problem.evaluate_solution(offspring[i]) for i in range(offspring.shape[0])])
        population = np.copy(offspring[costs.argsort()][:population_size])

        min_cost = np.min(costs)
        if global_min_cost > min_cost:
            global_best = population[0]
            global_min_cost = min_cost
            curr_steady_count = 0

    problem.plot(global_best, title="Population-based - Best solution")
    print(f"Final solution cost: {global_min_cost:,}")
    print(f"Generations: {generations}")
    print(f"Total evaluations: {problem.eval_cnt:,}")

    return global_best

def tsp_hillclimber(problem, steady_state):
    problem.eval_cnt = 0
    # Generate random starting solution
    solution = np.array(range(NUM_CITIES))
    np.random.shuffle(solution)
    solution_cost = problem.evaluate_solution(solution)

    problem.plot(solution, title="Hillclimber - initial")
    print(f"Initial solution cost: {solution_cost:,}")

    history = [(0, solution_cost)]
    curr_steady_count = 0
    step = 0
    while curr_steady_count < steady_state:
        step += 1
        curr_steady_count += 1
        new_solution = tweak_randomly(solution, pm=.5)
        new_solution_cost = problem.evaluate_solution(new_solution)
        if new_solution_cost < solution_cost:
            solution = new_solution
            solution_cost = new_solution_cost
            history.append((step, solution_cost))
            curr_steady_count = 0

    problem.plot(solution, title="Hillclimber - best solution")
    print(f"Final solution cost: {solution_cost:,}")
    print(f"Steps: {step}")
    print(f"Total evaluations: {problem.eval_cnt}")

    return solution

def main():
    problem = Tsp(NUM_CITIES)

    # Basic hillclimber
    print("Hillclimber TSP solver")
    solution = tsp_hillclimber(problem, steady_state=1000)
    solution_cost = problem.evaluate_solution(solution)
    print(f"Hillclimber solution cost: {solution_cost:,}")
    print("---------------------------\n")

    # input("Press a key to continue...")

    # Population based Crossover plus mutation
    print("Population-based (Crossover plus mutation) TSP solver")
    solution = tsp_population_based_solver(problem, population_size=20, offspring_size=50, steady_state=100, pm=0.2, px=0.3)
    solution_cost = problem.evaluate_solution(solution)
    print(f"Population-based (Crossover plus mutation) solution cost: {solution_cost:,}")
    print("")


if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().setLevel(level=logging.INFO)
    main()
    plt.show()