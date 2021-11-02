import torch
import torch.nn as nn

class NodeDownsampler:
    """
    This class provides the implementation for
    getting the graph down from its starting
    nodes down to a set number. See the README for
    more details

    Since the graph might be large, the edges are of the form:
    {start_node: {end_node}}
    """
    def __init__(self, node_dim, num_nodes_at_end):
        self.node_dim = node_dim
        self.num_nodes_at_end = num_nodes_at_end
        self.downsampler = nn.Linear(2*node_dim, node_dim)

    def downsample(self, nodes, edges):
        """
        Returns the nodes that were downsampled in the form of
        [(start_node_1, end_node_1), ..., (start_node_k, end_node_k)]
        """
        pass
