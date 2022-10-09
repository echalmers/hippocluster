from hippocluster.graphs.lfr import RandomWalkLFR
from hippocluster.algorithms.hippocluster import Hippocluster
import matplotlib.pyplot as plt
import numpy as np
import random
import itertools


plt.ion()


# create a small LFR graph
graph = RandomWalkLFR(n=100, tau1=2, tau2=1.1, mu=0.1, min_degree=3, max_degree=5, min_community=15, max_community=20)
colors = np.random.rand(100, 3)  # random colors for visualizing graph clusters
print(graph.n_clusters)

# instantiate Hippocluster object
hippocluster = Hippocluster(
    n_clusters=graph.n_clusters,
    drop_threshold=0.001
)

# perform clustering
for step in range(100):

    # get a batch of random walks
    walks = [
        set(graph.unweighted_random_walk(length=random.randint(15, 25)))
        for _ in range(graph.n_clusters*5 if step == 0 else graph.n_clusters)
    ]

    # update the clustering
    hippocluster.update(walks)
    assignments = hippocluster.get_assignments(graph)

    # plot original graph, random walks from each batch, and cluster assignments
    plt.clf()
    plt.subplot(1, 3, 1)
    graph.plot(node_colors={
        graph.nodes[i]: colors[graph.communities[i]] for i in range(graph.n_nodes)})
    plt.title('Sample LFR graph, showing ground truth clusters')

    plt.subplot(1, 3, 2)
    graph.plot(node_colors={node: colors[i] for i in range(len(walks)) for node in walks[i]})
    plt.title('random walks in this batch')

    plt.subplot(1, 3, 3)
    graph.plot(node_colors={node: colors[cluster][::-1] for node, cluster in assignments.items()})
    plt.title('Clustering found by Hippocluster')
    plt.pause(0.2)
