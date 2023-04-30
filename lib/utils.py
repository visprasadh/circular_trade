# Import libraries

import pandas as pd
import networkx as nx

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