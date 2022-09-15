import random
import math

from hippocluster.graphs.abstract import RandomWalkGraph
from hippocluster.algorithms.abstract import GraphClusteringAlgorithm

from scipy import sparse
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.cluster import kmeans_plusplus


class Hippocluster(GraphClusteringAlgorithm):
    """
    Implementation of the Hippocluster algorithm.
    Hippocluster performs graph clustering using a version of spherical k-means applied in the random-walk space
    """

    def __init__(self, n_clusters, batch_size, max_len, min_len=None, n_walks=None, lr=0.05, drop_threshold=0.02):
        """
        :param n_clusters: number of clusters to form
        :param batch_size: number of random walks to process at a time
        :param max_len: maximum walk length
        :param min_len: minimum walk length
        :param n_walks: total number of walks to use (should be a multiple of the graph size)
        :param lr: learning rate
        :param drop_threshold: minimum weight - weights below this threshold will be dropped
        """
        self.n_clusters = n_clusters
        self.lr = lr
        self.batch_size = batch_size
        self.max_len = max_len
        self.min_len = min_len
        self.map = dict()
        self.n_walks = int(n_walks) if n_walks is not None else None
        self.centers = None
        self.drop_threshold = drop_threshold

    def update(self, g: RandomWalkGraph, lr=None) -> int:
        """
        update the clusters on a single batch of random walks
        :param g: the graph being clustered
        :param lr: (optional) learning rate parameter. If specified, overrides the object's lr property
        :return: number of float values stored in the weight matrix
        """

        lr = lr or self.lr
        eq_sample_count = 1 / lr - 1

        # get walks
        walks = [
            set(g.unweighted_random_walk(length=random.randint(self.min_len, self.max_len)))
            for _ in range(self.batch_size)
        ]

        # assign new nodes
        all_states = set().union(*[walk for walk in walks])
        unmapped = all_states - set(self.map)
        if len(unmapped) > 0:
            self.map.update(dict(zip(unmapped, range(len(self.map), len(self.map) + len(unmapped)))))
            if self.centers is not None:
                self.centers.resize(self.n_clusters, len(self.map))

        # convert walks to binary
        x = self._walks_to_matrix(walks)

        # initialize if this is the first iteration
        if self.centers is None:
            self.centers = sparse.csr_matrix(
                normalize(kmeans_plusplus(x, self.n_clusters)[0], norm='l2', axis=1, copy=False)
            )

        # compute distances (dot products)
        dots = self.centers.dot(x.T)
        winners = dots.argmax(axis=0).A1

        # compute data means by cluster
        onehot_labels = sparse.csr_matrix(
            (np.ones(self.batch_size), (winners, np.arange(self.batch_size))), shape=(self.n_clusters, self.batch_size)
        )
        cluster_means = onehot_labels.dot(x)
        cluster_counts = onehot_labels.sum(axis=1).A1
        row_indices, _ = cluster_means.nonzero()
        cluster_means.data /= cluster_counts[row_indices]

        # derive a learning rate for each cluster, based on number of samples
        lr_by_cluster = cluster_counts / (eq_sample_count + cluster_counts)

        # do weighted average
        cluster_means.data *= lr_by_cluster[row_indices]
        row_indices, _ = self.centers.nonzero()
        self.centers.data *= (1 - lr_by_cluster)[row_indices]
        self.centers += cluster_means

        # prune
        self.centers.data[self.centers.data < self.drop_threshold] = 0
        self.centers.eliminate_zeros()

        # reproject cluster centers
        self.centers = normalize(self.centers, norm='l2', axis=1, copy=False)

        return len(self.centers.data)

    def _walks_to_matrix(self, walks):
        x = sparse.lil_matrix((len(walks), len(self.map)))
        for i in range(len(walks)):
            walk_set = set(walks[i])
            place_idx = list(map(self.map.get, walk_set))
            x[i, place_idx] = 1/math.sqrt(len(walk_set))
        return x

    def fit(self, g: RandomWalkGraph) -> dict:
        """
        perform clustering on the given graph
        :param g: graph to cluster
        :return: dictionary containing assignments, size (number of values stored in weight matrix), and walks used
        """
        steps = int(self.n_walks / self.batch_size)
        lr_sched = np.linspace(0.5, 0.01, steps)
        max_size = 0
        for i in range(steps):
            max_size = max(max_size, self.update(g, lr=lr_sched[i]))

        assignments = self.get_assignments(g)

        return {
            'assignments': assignments,
            'size': max_size,
            'walks': self.n_walks,
        }

    def get_assignments(self, g: RandomWalkGraph) -> dict:
        """
        assign each graph node to a cluster
        :param g: graph
        :return: dictionary of node -> cluster number
        """
        predictions = self.centers.argmax(axis=0).A1
        predictions[self.centers.sum(axis=0).A1 == 0] = -1
        assigns = {node: predictions[self.map[node]] for node in set(g.nodes).intersection(set(self.map))}
        assigns = {key: val for key, val in assigns.items() if val != -1}
        return assigns
