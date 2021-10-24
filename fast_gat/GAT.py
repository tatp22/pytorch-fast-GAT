import torch
import torch.nn as nn

class GraphAttentionHead(nn.Module):
    """
    A single head for the graph attention network.
    Currently, this implementation is slow because of the
    for loop.
    """
    def __init__(self, input_dim, output_dim, slope=0.2):
        super(GraphAttentionHead, self).__init__()
        self.w = nn.Linear(input_dim, output_dim)
        self.a = nn.Linear(2*output_dim, 1)
        self.activation = nn.LeakyReLU(slope)

    def forward(self, nodes, edges):
        nodes = self.w(nodes)
        new_nodes = nodes
        for curr_node, neighbors in edges.items():
            neighbors.extend([curr_node]) # Don't forget self attention!
            top_list = torch.tensor([self._get_top(nodes, curr_node, neighbor) for neighbor in neighbors])
            new_nodes[curr_node] = self._get_new_node_info(nodes, neighbors, top_list)
        return new_nodes

    def _get_top(self, nodes, i, j):
        """
        Returns a scalar for the top of the a_ij function
        nodes: Info of the nodes after the W param matrix
        i,j: Nodes to consider
        """
        cat = self.a(torch.cat((nodes[i], nodes[j]), dim=-1))
        x = torch.exp(self.activation(cat))
        return x

    def _get_new_node_info(self, nodes, neighbors, top_list):
        """
        nodes: List of nodes after the conv
        neighbors: list of which nodes are the neighbors of the current one
        top_list: list of a_ij from the function
        """
        bottom = torch.sum(top_list)
        x = torch.stack([top_list[curr_neighbor_no] * nodes[curr_neighbor] for curr_neighbor_no, curr_neighbor in enumerate(neighbors)])/bottom
        return torch.sum(x)

class GraphAttentionLayer(nn.Module):
    """
    A layer of the Graph Attention Network.
    NOTE! This is a different implementation than the original
    paper. Instead of averaging, this applies a learnable layer of
    size KFxF, more closely resembling the original attention paper.
    """
    def __init__(self, heads, input_dim, intermediate_dim, output_dim):
        super(GraphAttentionLayer, self).__init__()
        self.heads = nn.ModuleList()
        self.final_layer = nn.Linear(heads*intermediate_dim, output_dim)
        for head_no in range(heads):
            self.heads.extend([GraphAttentionHead(input_dim, intermediate_dim)])

    def forward(self, nodes, edges):
        head_outputs = []
        for head in self.heads:
            head_outputs.append(head(nodes, edges))
        out = torch.cat(head_outputs, dim=-1)
        out = self.final_layer(out).softmax(dim=-1)
        return out

class GraphAttentionNetwork(nn.Module):
    """
    The entrance for the Graph attention Network.
    This assumes that the inputs are:
    nodes: An NxF array, with N nodes and F dimensions for the features
    edges: A dictonary that represents directed edges. In particular, it has the form
    {x: [y,z]}
    which means that x has an edge to y and z. The edges do not repeat.
    Also, the edges are expected to be 0 indexed and in integer form.

    This function just returns the nodes, as the edges are unchanged.
    """
    def __init__(self, depth, heads, input_dim, inner_dim):
        super(GraphAttentionNetwork, self).__init__()
        self.layers = nn.ModuleList()
        if depth == 1:
            inner_dim = input_dim

        for layer_no in range(depth):
            if layer_no == 0:
                self.layers.extend([GraphAttentionLayer(heads, input_dim, inner_dim, inner_dim)])
            elif layer_no == depth-1:
                self.layers.extend([GraphAttentionLayer(heads, inner_dim, inner_dim, input_dim)])
            else:
                self.layers.extend([GraphAttentionLayer(heads, inner_dim, inner_dim, inner_dim)])

    def forward(self, nodes, edges):
        for layer in self.layers:
            nodes = layer(nodes, edges)
        return nodes
