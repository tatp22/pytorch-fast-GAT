import torch
import torch.nn as nn

class NodeUpsampler:
    """
    This class provides the implementation for
    the node upsampling back to the original size.
    """
    def __init__(self, node_dim, starting_nodes):
        self.node_dim = node_dim
        self.starting_nodes = starting_nodes
        self.upsampler = nn.Linear(node_dim, 2*node_dim)

    def upsample(self, dowsampled_nodes, downsample_order):
        """
        Returns only the upsampled nodes, assuming downsampled_order is of the form
        [(start_node_1, end_node_1), ..., (start_node_k, end_node_k)]
        """
        pass
