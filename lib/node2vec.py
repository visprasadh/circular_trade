from collections import defaultdict

import gensim
import numpy as np

from .node2vec_utils import *
from .utils import *
from .node2vec_utils import normalise


class Node2Vec:

    def __init__(self, graph, dim, walk_len, n_walks):

        self.graph = graph
        self.dim = dim
        self.walk_length = walk_len
        self.num_walks = n_walks
        self.p = 1.0
        self.q = 1.0
        self.d_graph = defaultdict(dict)
        self.strategy = {}

        self.compute_probabilities()
        self.walks = self.perform_walks()

    def compute_probabilities(self):
        
        # Compute the transition probabilities

        d_graph = self.d_graph

        for src in self.graph.nodes():

            # Init probabilities dict for first travel
            if 'probabilities' not in d_graph[src]:
                d_graph[src]['probabilities'] = dict()

            for current_node in self.graph.neighbors(src):

                # Init probabilities dict
                if 'probabilities' not in d_graph[current_node]:
                    d_graph[current_node]['probabilities'] = dict()

                wgts = list()
                d_neighbors = list()

                for dest in self.graph.neighbors(current_node):
                    
                    if current_node in self.strategy:
                        p = self.strategy[current_node].get('p', self.p)
                        q = self.strategy[current_node].get('q', self.q)
                    else:
                        p = self.p
                        q = self.q

                    try:
                        weight = fetch_weight(self.graph, current_node, dest) 
                    except:
                        weight = 1 

                    wgt = calc_weight(self.graph, src, dest, weight, p, q)

                    wgts.append(wgt)
                    d_neighbors.append(dest)

                d_graph[current_node]['probabilities'][src] = normalise(wgts)

            # Calculate first_travel weights for src
            d_graph[src]['first_travel_key'] = generate_first_travel_weights(self.graph, src)

            d_graph[src]['neighbors'] = list(self.graph.neighbors(src))

    def perform_walks(self):
        return gen_walks(self.d_graph, self.walk_length, self.num_walks, self.strategy)

    def fit(self, **skip_gram_params):

        # Function to create the embeddings using word2vec

        skip_gram_params['workers'] = 1
        skip_gram_params['vector_size'] = self.dim
        skip_gram_params['sg'] = 1

        return gensim.models.Word2Vec(self.walks, **skip_gram_params)
    
    
    