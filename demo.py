from hippocluster.graphs.lfr import RandomWalkLFR
from hippocluster.graphs.lattice import RandomWalkLattice
from hippocluster.algorithms.hippocluster import Hippocluster
from hippocluster.graphs.abstract import RandomWalkGraph
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    # create a small LFR graph
    graph = RandomWalkLFR(n=200, tau1=2, tau2=1.1, mu=0.1, min_degree=2, max_degree=5, min_community=20, max_community=50)

    # choose initial number of clusters
    N_CLUSTERS = graph.n_communities

    # instantiate Hippocluster object
    hippocluster = Hippocluster(
        lr=0.05,
        n_clusters=N_CLUSTERS,
        batch_size=50,
        max_len=int(graph.n_nodes / N_CLUSTERS * 1.25),
        min_len=int(graph.n_nodes / N_CLUSTERS * 0.75),
        n_walks=graph.n_nodes * 5,
        drop_threshold=0.00001,
        n_jobs=1
    )
    print(f'instantiating hippocluster with {hippocluster.n_clusters} clusters')

    # perform clustering
    import time
    start = time.time()
    hippocluster.fit(graph)
    assignments = hippocluster.get_assignments(graph)
    print(f'elapsed time: {time.time() - start:.2f}s')

    print(f'hippocluster found {len(set(assignments.values()))} clusters')

    # plot cluster assignments
    colors = np.random.rand(N_CLUSTERS*100, 3)
    graph.plot(node_colors={node: colors[cluster] for node, cluster in assignments.items()})
    plt.title('Clustering found by Hippocluster')
    plt.show()
