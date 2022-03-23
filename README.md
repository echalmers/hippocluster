# hippocluster
Hippocluster: an efficient, brain-inspired adaptation of K-means for graph clustering

## installation
you can install using pip:
`pip install git+https://github.com/echalmers/hippocluster.git`

## use
see [demo.py](https://github.com/echalmers/hippocluster/blob/master/demo.py) for basic usage examples

### creating graph objects for use with Hippocluster
Hippocluster's *fit* and *update* methods accept a [RandomWalkGraph](https://github.com/echalmers/hippocluster/blob/3759cdae5d449f5c32f9bb703ceb394d4a21929a/hippocluster/graphs/abstract.py#L13) object. There are two ways to make one of these from a NetworkX graph object. The first is to extend RandomWalkGraph, implementing *communities* and *n_communities*. Or if the graph has no ground truth communities, you can directly instantiate a RandomWalkGraph object with a NetworkX graph. For example:
`myGraph = RandomWalkGraph(nx.grid_2d_graph(20, 20))`
