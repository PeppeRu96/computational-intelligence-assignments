from evaluation import *
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
import matplotlib.pyplot as plt

# Pay attention: the player inside the move node is NOT the player who did the move inside the node!
# The player inside a given MoveNode is the player who has to choose the next move from there!
class MoveNode:
    def __init__(self, move, player, initial_node=False):
        self.move = move
        self.visits = 0
        self.reward = 0
        self.player = player
        self.initial_node = initial_node

    def q(self):
        return self.reward

    def n(self):
        return self.visits

def untried_actions(board, G, node):
    """
    Given a board, a graph, and a node, return the untried actions
    :param board:
    :param G:
    :param node:
    :return:
    """
    possible_moves = valid_moves(board)
    for c in G.successors(node):
        possible_moves.remove(c.move)

    return possible_moves

def is_fully_expanded(board, G, node):
    return len(untried_actions(board, G, node)) == 0

def is_terminal_node(board):
    return check_winner(board) != 0 or len(valid_moves(board)) == 0

def _best_child(G, node, c_param=0.1):
    # Understanding Q:
    # reward positive * player 1 -> positive: correct!
    # reward negative * player 1 -> negative: correct!
    # reward positive * player -1 -> negative: correct!
    # reward negative * player -1 -> positive: correct!
    successors = list(G.successors(node))
    nodes_weights = [(c.q() * node.player / c.n()) + c_param * np.sqrt((2 * np.log(node.n()) / c.n())) for c in successors]
    return successors[np.argmax(nodes_weights)]

def expand(board, G, node):
    untried_moves = untried_actions(board, G, node)

    m = np.random.choice(untried_moves)
    n = MoveNode(m, - node.player)
    G.add_node(n)
    G.add_edge(node, n, color='black')

    return n

def rollout_random_sampling(board, node, samples):
    reward = random_rollout_estimate(board, node.player, samples)
    return reward

def rollout_static_heuristic_magic_square(board):
    reward = heuristic_static_magic_square_estimate(board)
    return reward

def backpropagate(G, node, reward):
    node.visits += 1
    node.reward += reward
    predecessors = list(G.predecessors(node))
    if len(predecessors) > 0 and not node.initial_node:
        backpropagate(G, predecessors[0], reward)

def select(board, G, node, c_param=0.1):
    while not is_terminal_node(board):
        if not is_fully_expanded(board, G, node):
            child = expand(board, G, node)
            play(board, child.move, node.player)
            return child
        else:
            best_node = _best_child(G, node, c_param=c_param)
            play(board, best_node.move, node.player)
            node = best_node
    return node

def _get_iterator(iterations = None, max_time = None):
    """
    It allows to stop montecarlo based on iterations or time
    :param iterations:
    :param max_time:
    :return:
    """
    if iterations is not None:
        def f():
            for i in range(iterations):
                yield i
        return f
    else:
        def f():
            start = timeit.default_timer()
            i = 0
            while True:
                yield i
                end = timeit.default_timer()
                if (end-start) > max_time:
                    break
                i += 1
        return f

def draw_graph(G, title, show_only_subgraph=False):
    if show_only_subgraph:
        init_node = [c for c in G.nodes if c.initial_node == True][0]
        all_childs = nx.descendants(G, init_node)
        G = G.subgraph([init_node, *all_childs])

    #labels = {node: (f"{node.q()/node.n():.1f}" if node.n() != 0 else f"{node.q():.1f}") for node in G.nodes}
    labels = {node: (f"Q:{node.q()/node.n():.1f} - R: {node.reward:.2f}" if node.n() != 0 else f"{node.q():.1f}") for node in G.nodes}

    edge_labels = {(n1, n2): f"{n2.move + 1}" for (n1, n2) in G.edges}
    edge_colors = [G[u][v]['color'] for u, v in G.edges]
    node_colors = []
    for node in G.nodes:
        if node.player == 1:
            color = 'red'
        else:
            color = 'blue'
        if node.initial_node:
            color = 'green'
        node_colors.append(color)
    pos = graphviz_layout(G, prog="dot")
    ax = plt.gca()
    ax.set_title(title)
    nx.draw(G, pos, labels = labels, ax=ax, node_color=node_colors, edge_color=edge_colors, font_size=10, node_size=600)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.show()

def _montecarlo_tree_search(board, player, G:nx.DiGraph, start_node:MoveNode = None, iterations = None, max_time = None, rollout_f=None, c_param=0.1):
    iterator = _get_iterator(iterations, max_time)

    if G is None:
        G = nx.DiGraph()

    if start_node is None:
        start_node = MoveNode(-1, player, initial_node=True)
        G.add_node(start_node)

    for i in iterator():
        board_copy = board.copy()
        node = select(board_copy, G, start_node, c_param)
        reward = rollout_f(board_copy, node)
        backpropagate(G, node, reward)

    return start_node

def montecarlo_tree_search(board, player, G:nx.DiGraph, start_node:MoveNode = None, iterations = None, max_time = None, rollout_f=None, draw_tree=False, draw_only_sub_tree=False, c_param=0.1):
    # if is_terminal_node(board):
    #     if start_node is None:
    #         start_node = MoveNode(None, player, initial_node=True)
    #         start_node.reward = check_winner(board)
    #         G.add_node(start_node)
    #     else:
    #         start_node.reward = check_winner(board)
    #     return start_node

    # Standard montecarlo
    start_node = _montecarlo_tree_search(board, player, G, start_node, iterations, max_time, rollout_f, c_param)
    best_node = _best_child(G, start_node, c_param=0.)
    G[start_node][best_node]['color'] = 'blue'
    if draw_tree:
        draw_graph(G, f"Player: {start_node.player} - Best move: {best_node.move + 1} - Q/n: {(best_node.q()/best_node.n() if best_node.n() != 0 else best_node.q()):.1f} - P: 1 (red) - P: -1 (blue)", show_only_subgraph=draw_only_sub_tree)
    start_node.initial_node = False

    return best_node

class MCTS:
    def __init__(self, player_n, G : nx.DiGraph = None, iterations = None, max_time = None, rollout_f = None, draw_tree=False, draw_only_sub_tree=False, c_param=0.1):
        self.G = G
        if self.G is None:
            self.G = nx.DiGraph()
        self.iterations = iterations
        self.max_time = max_time
        self.rollout_f = rollout_f
        self.current_node = None
        self.player_n = player_n
        self.draw_tree = draw_tree
        self.draw_only_sub_tree = draw_only_sub_tree
        self.c_param = c_param

    def play(self, board, turn):
        best_node = montecarlo_tree_search(board, turn, self.G, self.current_node, self.iterations, self.max_time, self.rollout_f, self.draw_tree, self.draw_only_sub_tree, self.c_param)
        self.current_node = best_node
        return self.current_node.move

    def set_opponent_move(self, board, turn, move):
        if self.player_n == turn:
            return None

        # The opponent started, we create a node for him
        if self.current_node is None:
            node_opponent = MoveNode(-1, turn, initial_node=False)
            self.G.add_node(node_opponent)
            node = MoveNode(move, self.player_n, initial_node=True)
            self.G.add_edge(node_opponent, node, color='red')
            self.current_node = node
            return None

        # Set the edge to red and current_node forward
        for c in self.G.successors(self.current_node):
            if c.move == move:
                self.G[self.current_node][c]['color'] = 'red'
                self.current_node = c
                self.current_node.initial_node = True
                return None

        node = MoveNode(move, self.player_n, initial_node=True)
        self.G.add_node(node)
        self.G.add_edge(self.current_node, node, color='red')

        self.current_node = node