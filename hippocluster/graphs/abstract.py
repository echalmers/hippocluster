from abc import abstractmethod
import itertools
import random
import time
from warnings import warn

import networkx as nx
from networkx.algorithms.community.quality import modularity as nx_modularity
from scipy import sparse
import numpy as np
from sklearn.metrics import normalized_mutual_info_score


class RandomWalkGraph:
    """
    composition class that extends a networkx graph with the ability to generate random walks, and other functionality
    """

    def __init__(self, networkx_graph, pos=None, walk_type='walk'):
        """
        :param networkx_graph: base networkx graph to use
        :parm pos: a networkx layout for plotting
        """
        self.G = networkx_graph
        self.pos = pos
        self.shuffled_node_list = None
        self.random_walk_index = 0
        self.adj = None

        if walk_type == 'diffusion':
            if not hasattr(networkx_graph, 'successors'):
                raise ValueError("diffusion walk_type only works with directed graphs that have successors and predecessors methods")
            self._walk_generator = self._random_diffusion
        else:
            self._walk_generator = self._random_walk

    @property
    def n_nodes(self):
        return len(self.G.nodes)

    @property
    def nodes(self):
        return list(self.G.nodes)

    @property
    def edges(self):
        return list(self.G.edges)

    @property
    @abstractmethod
    def communities(self):
        """
        should return an iterable of class assignments for each of the graph nodes.
        if the graph has no ground truth communities, return None
        """
        return None

    @property
    @abstractmethod
    def n_communities(self):
        """
        should return the integer number of ground truth communities for this graph
        if the graph has no ground truth communities, return None
        """
        return None

    def modularity(self, cluster_assignments: dict = None):
        """
        compute the modularity of a clustering on this graph
        :param cluster_assignments: dict of node -> cluster number
        :return: modularity score
        """
        if cluster_assignments is not None:
            communities = {assign: set() for assign in set(cluster_assignments.values())}
            for node, assign in cluster_assignments.items():
                communities[assign].add(node)
            communities = list(communities.values())
            for node in set(self.nodes) - set(itertools.chain.from_iterable(communities)):
                communities.append({node})
        else:
            communities = {assign: set() for assign in set(self.communities)}
            for i in range(len(self.nodes)):
                communities[self.communities[i]].add(self.nodes[i])
            communities = list(communities.values())

        return nx_modularity(
            G=self.G,
            communities=communities
        )

    @property
    def adjacency_matrix(self):
        if self.adj is None:
            self.adj = nx.adjacency_matrix(self.G).tocsr()
        return self.adj

    def unweighted_random_walk(self, length):
        """
        generate a random walk without considering edge weights
        :param length: maximum walk length
        :param n: number of walks to generate. If n>1, return will be a list of walk-lists
        :return: list of nodes passed
        """
        if self.shuffled_node_list is None:
            self.shuffled_node_list = list(self.G)
            random.shuffle(self.shuffled_node_list)

        # nodes = [start_node or random.choice(list(self.G.nodes))]
        nodes = [self.shuffled_node_list[self.random_walk_index]]
        self.random_walk_index = (self.random_walk_index + 1) % len(self.shuffled_node_list)

        for step in range(length - 1):
            options = self.G.adj.get(nodes[-1], [])
            if len(options) == 0:
                return nodes
            nodes.append(random.choice(list(options)))

        return nodes

    def unweighted_random_walks(self, min_length, max_length, n):
        warn('unweighted_random_walks will be deprecated in a future version. use random_walks instead')
        return [
            self.unweighted_random_walk(length=random.randint(min_length, max_length))
            for _ in range(n)
        ]

    def random_walk(self, length):
        """
        generate a random walk, considering edge weights
        :param length: maximum walk length
        :param n: number of walks to generate. If n>1, return will be a list of walk-lists
        :return: list of nodes passed
        """
        if self.shuffled_node_list is None:
            self.shuffled_node_list = list(self.G)
            random.shuffle(self.shuffled_node_list)

        # nodes = [start_node or random.choice(list(self.G.nodes))]
        start_node = self.shuffled_node_list[self.random_walk_index]
        self.random_walk_index = (self.random_walk_index + 1) % len(self.shuffled_node_list)

        return self._walk_generator(length, start_node)

    def _random_walk(self, length, start_node):

        nodes = [start_node]

        for step in range(length - 1):
            options = self.G.adj.get(nodes[-1], dict())
            if len(options) == 0:
                return nodes
            weights = [options[neighbor].get('weight', 1) for neighbor in options]
            nodes.extend(random.choices(list(options), weights=weights, k=1))

        return nodes

    def _random_diffusion(self, size, start_node):
        """
        generate a set of nodes by growing a frontier of nodes from the start state.
        grow this frontier backward if the start_node is a sink
        :param size: quit when the set of nodes reaches this size
        :return: set of nodes
        """

        nodes = [start_node]
        frontier = dict()
        forward = len(self.G[start_node]) > 0

        for step in range(size - 1):
            if forward:
                for node in self.G.successors(nodes[-1]):
                    weight = self.G[nodes[-1]][node].get('weight', 1.0)
                    frontier[node] = frontier.get(node, 0.0) + weight
            else:
                for node in self.G.predecessors(nodes[-1]):
                    weight = self.G[node][nodes[-1]].get('weight', 1.0)
                    frontier[node] = frontier.get(node, 0.0) + weight

            if len(frontier) == 0:
                return nodes
            nodes.extend(random.choices(list(frontier), weights=frontier.values(), k=1))
            frontier.pop(nodes[-1], None)

        return nodes

    def random_walks(self, min_length, max_length, n, weighted=True):
        if weighted:
            return [
                self.random_walk(length=random.randint(min_length, max_length))
                for _ in range(n)
            ]
        else:
            return [
                self.unweighted_random_walk(length=random.randint(min_length, max_length))
                for _ in range(n)
            ]

    def restrained_unweighted_random_walk(self, max_l, w, th, return_set=False, start_node=None):
        """
        generate a restrained random walk using the method of Okuda et al in “Communi-ty Detection Using Restrained
        Random-Walk Similarity,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 43, no. 1, pp. 89–103, 2019
        :param max_l: max walk length
        :param w: window size
        :param th: threshold
        :param return_set: set True to return the unique set of nodes passed. False returns a list of nodes passed
                        (which may include duplicates)
        :param start_node: optional start node
        :return: set of list of nodes passed
        """
        nodes = [start_node or random.choice(list(self.G.nodes))]
        unique = set(nodes)
        n = [1]

        for step in range(max_l - 1):
            options = self.G.adj.get(nodes[-1])
            new = random.choice(list(options))
            nodes.append(new)
            unique.add(new)
            n.append(len(unique))

            if len(nodes) > w:
                if n[-1] - n[-w] < th:
                    break

        if return_set:
            return unique
        return nodes

    def plot(self, node_colors: dict = None, edge_colors=None, edgewidth: int = 1):
        """
        plot graph
        :param node_colors: dict of node -> color vector
        :param edge_colors: dict of edge -> color vector
        :param edgewidth: integer edge width
        :return: None
        """
        if node_colors is None:
            node_colors = {}
        if edge_colors is None:
            edge_colors = {}
        self.pos = self.pos or nx.spring_layout(self.G, seed=42)
        nx.draw(self.G, self.pos, with_labels=False,
                node_color=[node_colors.get(node, [0, 0, 0]) for node in self.G.nodes],
                edge_color=[edge_colors.get(edge, [0, 0, 0]) for edge in self.G.edges],
                node_size=400, width=edgewidth, font_size=15,
                )

    def score_clustering_method(self, algorithm) -> dict:
        """
        evaluate a clustering algorithm on this graph
        :param algorithm:
        :return: dict of metrics
        """
        start = time.time()
        metrics = algorithm.fit(self)
        score = normalized_mutual_info_score([metrics['assignments'].get(node, np.random.randint(10000, 1000000)) for node in self.nodes], self.communities)
        # print(algorithm, score, time.time() - start)
        metrics['score'] = score
        metrics['fit_time'] = time.time() - start
        return metrics