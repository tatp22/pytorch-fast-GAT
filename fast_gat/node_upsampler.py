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

    def upsample(self, downsampled_nodes, downsample_order, downsample_map):
        """
        :param downsampled_nodes: (starting_nodes, node_dim) array of node information
        :param downsample_order: [(source_1, dest_1), ..., (source_k, dest_k)] of how the nodes were merged
        :param downsample_map: A map from the (starting_nodes) to the original (n) number of nodes. All {0, ..., n-1} nodes will be a key here.
        :ret: The nodes back to their original size
        """
        full_size_mat = self.get_full_size_mat(downsampled_nodes, downsample_order, downsample_map)
        return self.fill_mat(full_size_mat, downsample_order)

    def get_full_size_mat(self, downsampled_nodes, downsample_order, downsample_map):
        full_size_mat = torch.zeros((self.starting_nodes + len(downsample_order), self.node_dim))
        for i, node in enumerate(downsampled_nodes):
            full_size_mat[downsample_map[i]] = node
        return full_size_mat

    def fill_mat(self, full_size_mat, downsample_order):
        for merged_node in downsample_order[::1]:
            source = merged_node[0]
            dest = merged_node[1]
            upsampled_node = self.upsampler(full_size_mat[source])
            full_size_mat[source], full_size_mat[dest] = upsampled_node[:self.node_dim], upsampled_node[self.node_dim:]
        return full_size_mat

