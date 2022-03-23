from abc import abstractmethod

from hippocluster.graphs.abstract import RandomWalkGraph


class GraphClusteringAlgorithm:

    @abstractmethod
    def fit(self, g: RandomWalkGraph) -> dict:
        """
        perform clustering on the given graph
        :param g: graph to cluster
        :return: dictionary containing assignments, size (number of values stored in weight matrix), and walks used
        """
        raise NotImplementedError()

    @abstractmethod
    def get_assignments(self, g: RandomWalkGraph) -> dict:
        """
        assign each graph node to a cluster
        :param g: graph
        :return: dictionary of node -> cluster number
        """
        raise NotImplementedError()
