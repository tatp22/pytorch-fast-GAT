import copy
import random
import torch
import torch.nn as nn

from disjoint_set import DisjointSet

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
        [(start_node_1, end_node_1), ..., (start_node_k, end_node_k)],
        as well as the nodes themselves
        """
        leaders = DisjointSet()
        n, d = nodes.size()
        nodes_copy = copy.deepcopy(nodes)
        edges_copy = copy.deepcopy(edges)
        out_nodes = torch.zeros((self.num_nodes_at_end, d))
        out_edges = list()
        while n - len(out_edges) > self.num_nodes_at_end:
            edge = self.choose_edge(edges_copy)
            source, dest = edge[0], leaders[edge[1]]
            nodes_copy[source] = self.downsampler(torch.cat((nodes_copy[source], nodes_copy[dest])))
            out_edges.append((source, dest))
            leaders.union(dest, source)
        inds_filled = 0
        for i, node in enumerate(nodes):
            if leaders.find(i) == i:
                out_nodes[inds_filled] = node
                inds_filled += 1
        assert inds_filled == self.num_nodes_at_end, "Downsampling failed! Only {} nodes were filled out of {}.".format(inds_filled, self.num_nodes_at_end)
        return out_nodes, out_edges

    def choose_edge(self, edges):
        """Currently chooses UAR"""
        start, end_set = random.choice(list(edges.items()))
        end = random.choice(list(end_set))
        edges[start].discard(end)
        if not len(edges[start]):
            del edges[start]
        return start, end
