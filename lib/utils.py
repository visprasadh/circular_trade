# Import libraries

import pandas as pd
import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt

from .node2vec import *
from .node2vec import Node2Vec
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE

# Utility functions

def import_data(path):
    # Create dataframe out of csv dataset
    df = pd.read_csv(path)
    # Convert dataframe into a list of tuples of format (source_vertex, destination_vertex, weight)
    data = list(df.itertuples(index = False, name = None))
    return data

def create_multi_graph(data):
    # Create a Multi Directed graph object
    multi_graph = nx.MultiDiGraph()
    # Add weighted edges and nodes from data
    multi_graph.add_weighted_edges_from(data)
    return multi_graph

def create_undirected_graph(multi_graph):
    # Create a graph object
    graph = nx.Graph()
    # Look at source, target vertices and weights for each edge
    for u,v,data in multi_graph.edges(data=True):
        # Provision if weight is not present in dataset
        w = data['weight'] if 'weight' in data else 1.0
        # If undirected graph edge is present, add current weight to weight already present
        if graph.has_edge(u,v):
            graph[u][v]['weight'] += w
        # Else create new edge in undirected graph with current weight
        else:
            graph.add_edge(u, v, weight=w)
    return graph

def generate_embeddings(graph):
    # Generate embeddings using node2vec
    node2vec = Node2Vec(graph, 64, 30, 200) 
    # Embed nodes
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    embeddings = []
    for node in list(graph.nodes()):
        node_embedding = model.wv[f'{node}']
        embeddings.append(node_embedding)
    return np.array(embeddings)

def dbs(embeddings, eps, min_samples):
    # Reducing the number of dimensions
    tsne = TSNE(n_components = 2, random_state = 0)
    tsne_data = tsne.fit_transform(embeddings)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(tsne_data)
    return tsne_data, clustering.labels_

def cluster_plot(tsne, clusters):
    cluster_data = pd.DataFrame()
    cluster_data['x'] = tsne[:, 0]
    cluster_data['y'] = tsne[:, 1]
    cluster_data['cluster'] = clusters
    plot = sns.scatterplot(data = cluster_data, x = 'x', y = 'y', hue='cluster')
    plot.legend_.remove()
    plt.show()