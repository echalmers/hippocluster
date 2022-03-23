from hippocluster.graphs.abstract import RandomWalkGraph

from networkx import LFR_benchmark_graph
import numpy as np


class RandomWalkLFR(RandomWalkGraph):
    """
    Lancichinetti–Fortunato–Radicchi benchmark network
    """

    def __init__(self, n, tau1, tau2, mu, min_degree, max_degree, min_community, max_community, seed=42, **kwargs):
        """
        Lancichinetti–Fortunato–Radicchi benchmark network
        :param n: number of nodes
        :param tau1: or "gamma", power law exponent for degree
        :param tau2: or "beta", power law exponent for community size
        :param mu: avg fraction of neighboring nodes that are outside the community
        :param min_degree:
        :param max_degree:
        :param min_community:
        :param max_community:
        :param seed:
        """
        self.params = locals()
        del self.params['self']

        G = LFR_benchmark_graph(n=n, tau1=tau1, tau2=tau2, mu=mu, min_degree=min_degree, max_degree=max_degree,
                                min_community=min_community, max_community=max_community, seed=seed)
        super().__init__(G)
        self.classes = None
        self.n_clusters = None
        self.compute_communities()

    def compute_communities(self):
        node_names = list(self.G.nodes)
        all_communities = list({tuple(self.G.nodes[i]['community']) for i in self.G})
        self.classes = np.zeros(len(self.G.nodes)).astype(int)
        for i in range(len(all_communities)):
            for node_name in all_communities[i]:
                node_ind = node_names.index(node_name)
                self.classes[node_ind] = i
        self.n_clusters = len(np.unique(self.classes))

    @property
    def communities(self):
        return self.classes

    @property
    def n_communities(self):
        return self.n_clusters
