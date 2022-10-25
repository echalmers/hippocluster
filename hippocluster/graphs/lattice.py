from hippocluster.graphs.abstract import RandomWalkGraph

from networkx.generators import grid_2d_graph
from networkx import planar_layout


class RandomWalkLattice(RandomWalkGraph):

    def __init__(self, m, n):
        """
        2d lattice graph

        """
        g = grid_2d_graph(m, n)
        super().__init__(g, pos={node: list(node) for node in g.nodes})

    @property
    def communities(self):
        pass

    @property
    def n_communities(self):
        pass
