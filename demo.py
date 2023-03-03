from hippocluster.graphs.lfr import RandomWalkLFR
from hippocluster.graphs.lattice import RandomWalkLattice
from hippocluster.algorithms.hippocluster import Hippocluster
from hippocluster.graphs.abstract import RandomWalkGraph
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    # choose initial number of clusters
    N_CLUSTERS = 10

    # create a small LFR graph
    # graph = RandomWalkLFR(n=200, tau1=2, tau2=1.1, mu=0.1, min_degree=2, max_degree=5, min_community=20, max_community=50)
    # graph = RandomWalkLattice(n=10, m=10)
    g = nx.DiGraph(nx.generators.grid_2d_graph(10, 10))
    for edge in list(g.edges):
        if edge[0][0] < edge[1][0]:
            g.remove_edge(edge[0], edge[1])
        elif np.random.rand() < 0.1:
            g.remove_edge(edge[0], edge[1])
    graph = RandomWalkGraph(g)
    graph.pos = {node: node for node in g}


    if hasattr(graph, 'n_clusters'):
        print(f'graph has {graph.n_clusters} ground-truth clusters')

    # instantiate Hippocluster object
    hippocluster = Hippocluster(
        lr=0.05,
        n_clusters=N_CLUSTERS,
        batch_size=N_CLUSTERS*5,
        max_len=int(graph.n_nodes / N_CLUSTERS * 1.25),
        min_len=int(graph.n_nodes / N_CLUSTERS * 0.75),
        n_walks=graph.n_nodes * 5,
        drop_threshold=0.00001,
        n_jobs=3
    )
    print(f'instantiating hippocluster with {hippocluster.n_clusters} clusters')

    # perform clustering
    import time
    start = time.time()
    hippocluster.fit(graph)
    assignments = hippocluster.get_assignments(graph)
    print(time.time() - start)

    # from sknetwork.clustering import Louvain
    # c = Louvain()
    # labels = c.fit_transform(nx.adjacency_matrix(g))
    # assignments = {graph.nodes[i]: labels[i] for i in range(len(labels))}

    print(f'hippocluster found {len(set(assignments.values()))} clusters')

    # plot cluster assignments
    colors = np.random.rand(N_CLUSTERS*100, 3)
    graph.plot(node_colors={node: colors[cluster] for node, cluster in assignments.items()})
    plt.title('Clustering found by Hippocluster')
    plt.show()