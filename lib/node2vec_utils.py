import random

from .utils import *

def normalise(weights):
    weights = np.array(weights)
    return weights / weights.sum()

def calculate_length(src, strategy, global_len):
    return strategy[src].get('walk_length', global_len) if src in strategy else global_len

def calc_prob(g, walk):
    walk_len = len(walk)
    if walk_len == 1:
        return g[walk[-1]]['first_travel_key']
    else:
        return g[walk[-1]]['probabilities'][walk[-2]]
    
def calc_weight(g, src, dest, w, p, q):
    if dest == src:
        # If nodes are same, backwards probability
        return w * 1 / p
    elif dest in g[src]: 
        # If the source and the destination are connected
        return w
    else:
        # Otherwise
        return w * 1 / q

def perform_walk(walk, walk_len, graph):
    while len(walk) < walk_len:
        # Routes that can be taken during random walk
        routes = graph[walk[-1]].get('neighbors', None)
        # If the current node is a dead end and there are no further routes available
        if not routes:
            break
        # Calculate probabilities
        prob = calc_prob(graph, walk)
        # Walk destination
        dest = random.choices(routes, weights = prob)[0]
        walk.append(dest)
    return walk

def skip_condition(src, stg, n):
    c1 = src in stg
    c2 = 'num_walks' in stg[src] if c1 else False
    c3 = stg[src]['num_walks'] <= n if c2 else False
    return c1 and c2 and c3

def shuffle(l):
    l = list(l)
    random.shuffle(l)
    return l

def gen_walks(d_graph, global_walk_length, num_walks, sampling_strategy): 
    # Generates the random walks
    walks = list()

    for n_walk in range(num_walks):
        # Shuffle the nodes
        shuffled = shuffle(d_graph.keys())

        # Start a random walk from every node
        for source in shuffled:
            if skip_condition(source, sampling_strategy, n_walk):
                continue
            # Start walk
            walk = [source]
            # Calculate walk length
            walk_length = calculate_length(source, sampling_strategy, global_walk_length)
            walk = perform_walk(walk, walk_length, d_graph)
            walk = list(map(str, walk))  # Convert all to strings
            walks.append(walk)
    return walks

def generate_first_travel_weights(graph, src):
    weights = []
    for dest in graph.neighbors(src):
        weights.append(graph[src][dest].get('weight',1))
    weights = normalise(weights)
    return weights

def fetch_weight(g, curr, dest):
    if g[curr][dest].get('weight'):
        return g[curr][dest].get('weight', 1)
    else: 
        e = list(g[curr][dest])[-1]
        return g[curr][dest][e].get('weight', 1)  