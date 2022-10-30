from hippocluster.graphs.lfr import RandomWalkLFR
from hippocluster.graphs.lattice import RandomWalkLattice
from hippocluster.algorithms.hippocluster import Hippocluster
import matplotlib.pyplot as plt
import numpy as np

# choose initial number of clusters
N_CLUSTERS = 5

# create a small LFR graph
graph = RandomWalkLFR(n=200, tau1=2, tau2=1.1, mu=0.1, min_degree=2, max_degree=5, min_community=20, max_community=50)
# graph = RandomWalkLattice(n=10, m=10)

if hasattr(graph, 'n_clusters'):
    print(f'graph has {graph.n_clusters} ground-truth clusters')

# instantiate Hippocluster object
hippocluster = Hippocluster(
    lr=0.2,
    n_clusters=N_CLUSTERS,
    batch_size=N_CLUSTERS,
    max_len=75,
    min_len=20,
    n_walks=graph.n_nodes * 10,
    drop_threshold=0.001
)
print(f'instantiating hippocluster with {hippocluster.n_clusters} clusters')

# perform clustering
hippocluster.fit(graph)
assignments = hippocluster.get_assignments(graph)

print(f'hippocluster found {len(set(assignments.values()))} clusters')

# plot cluster assignments
colors = np.random.rand(N_CLUSTERS, 3)
graph.plot(node_colors={node: colors[cluster][::-1] for node, cluster in assignments.items()})
plt.title('Clustering found by Hippocluster')
plt.show()