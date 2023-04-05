from hippocluster.graphs.abstract import RandomWalkGraph
from hippocluster.algorithms.hippocluster import Hippocluster
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random


plt.ion()
random.seed(0)
np.random.seed(0)


# create a small sample graph
graph = RandomWalkGraph(nx.connected_caveman_graph(6, 10))

colors = [
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 0],
    [0, 1, 1],
    [1, 1, 0],
    [1, 0, 1],
    [0, 0, 0]]  # colors for visualizing graph clusters

# instantiate Hippocluster object
hippocluster = Hippocluster(
    n_clusters=6,
    drop_threshold=0.001,
    lr=0.1
)

# supply first batch of random walks to initialize hippocluster
hippocluster.update(
    walks=graph.random_walks(
        min_length=2,
        max_length=3,
        n=50
    ))

# now supply one walk at a time - visualizing as we go

for step in range(200):

    # get a batch of random walks
    walk = graph.random_walks(min_length=5, max_length=15, n=1)

    # update the clustering
    hippocluster.update(walk)
    assignments = hippocluster.get_assignments(graph)

    # plot original graph, random walk, and cluster assignments
    plt.clf()

    plt.subplot(1, 2, 1)
    graph.plot(node_colors={node: [0, 0, 0] if node not in walk[0] else [0.7, 0.7, 0.7] for node in graph.nodes})
    plt.title('random walk')

    plt.subplot(1, 2, 2)
    graph.plot(node_colors={node: colors[assignments.get(node, -1)] for node in graph.nodes})
    plt.title('Clustering found by Hippocluster')

    plt.pause(0.5 if step < 50 else 0.01)
