from hippocluster.graphs.lfr import RandomWalkLFR
from hippocluster.algorithms.hippocluster import Hippocluster
import matplotlib.pyplot as plt
import numpy as np


# create a small LFR graph
graph = RandomWalkLFR(n=200, tau1=2, tau2=1.1, mu=0.1, min_degree=2, max_degree=30, min_community=20, max_community=50)

# instantiate Hippocluster object
hippocluster = Hippocluster(
    n_clusters=graph.n_clusters,
    batch_size=graph.n_clusters*3,
    max_len=75,
    min_len=20,
    n_walks=graph.n_nodes * 10,
    drop_threshold=0.001
)

# perform clustering
hippocluster.fit(graph)
assignments = hippocluster.get_assignments(graph)

# plot original graph and cluster assignments
plt.subplot(1, 2, 1)
colors = np.random.rand(graph.n_clusters, 3)
graph.plot(node_colors={graph.nodes[i]: colors[graph.communities[i]] for i in range(graph.n_nodes)})
plt.title('Sample LFR graph, showing ground truth communities')
plt.subplot(1, 2, 2)
graph.plot(node_colors={node: colors[cluster][::-1] for node, cluster in assignments.items()})
plt.title('Clustering found by Hippocluster')

# measure normalized mutual information score for hippocluster on a range of graph sizes
n_values = [500, 1000, 2000, 4000]
nmi = []
for n in n_values:
    print(f'clustering LFR graph of size {n}')
    graph = RandomWalkLFR(
        n=n, tau1=2, tau2=1.1, mu=0.1, min_degree=2, max_degree=30, min_community=20, max_community=50
    )

    hippocluster = Hippocluster(
        n_clusters=graph.n_clusters,
        batch_size=graph.n_clusters * 3,
        max_len=75,
        min_len=20,
        n_walks=graph.n_nodes * 15,
        drop_threshold=0.02
    )

    nmi.append(graph.score_clustering_method(algorithm=hippocluster)['score'])

# plot NMI results
plt.figure()
plt.plot(n_values, nmi)
plt.ylim([0.6, 1])
plt.ylabel('normalized mutual information score')
plt.xlabel('graph size')
plt.show()
