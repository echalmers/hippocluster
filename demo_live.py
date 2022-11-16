from hippocluster.graphs.lfr import RandomWalkLFR
from hippocluster.graphs.lattice import RandomWalkLattice
from hippocluster.algorithms.hippocluster import Hippocluster
import matplotlib.pyplot as plt
import numpy as np
import random


plt.ion()


# create a small sample graph
# graph = RandomWalkLFR(n=100, tau1=2, tau2=1.1, mu=0.1, min_degree=3, max_degree=5, min_community=15, max_community=20)
graph = RandomWalkLattice(m=10, n=10)
colors = np.random.rand(100, 3)  # random colors for visualizing graph clusters

# set a desired number of clusters
N_CLUSTERS = graph.n_communities or 5

# instantiate Hippocluster object
hippocluster = Hippocluster(
    n_clusters=N_CLUSTERS,
    drop_threshold=0.001
)

# perform clustering
for step in range(100):

    # get a batch of random walks
    walks = graph.unweighted_random_walks(min_length=15, max_length=25, n=N_CLUSTERS*5 if step == 0 else N_CLUSTERS)

    # update the clustering
    hippocluster.update(walks)
    assignments = hippocluster.get_assignments(graph)


    # plot original graph, random walks from each batch, and cluster assignments
    plt.clf()
    n_panels = 2 if graph.n_communities is None else 3

    plt.subplot(1, n_panels, 1)
    graph.plot(node_colors={node: colors[i] for i in range(len(walks)) for node in walks[i]})
    plt.title('random walks in this batch')

    plt.subplot(1, n_panels, 2)
    graph.plot(node_colors={node: colors[cluster][::-1] for node, cluster in assignments.items()})
    plt.title('Clustering found by Hippocluster')

    if graph.n_communities is not None:
        plt.subplot(1, n_panels, 3)
        graph.plot(node_colors={graph.nodes[i]: colors[graph.communities[i]] for i in range(graph.n_nodes)})
        plt.title('ground truth clusters')
    plt.pause(0.2)
