import torch
import torch.nn as nn

class DenseGraphAttentionHead(nn.Module):
    """
    A graph attention head.
    Speedup techniques taken from:
    https://github.com/Diego999/pyGAT/blob/master/layers.py
    """
    def __init__(self, input_dim, output_dim):
        super(DenseGraphAttentionHead, self).__init__()

    def forward(self, nodes, edge_mat):
        pass #TODO!

class DenseGraphAttentionLayer(nn.Module):
    """
    A dense layer of the graph attention network.
    """
    def __init__(self, heads, input_dim, intermediate_dim, output_dim):
        super(DenseGraphAttentionLayer, self).__init__()
        self.heads = nn.ModuleList()
        self.final_layer = nn.Linear(heads*intermediate_dim, output_dim)
        for head in range(heads):
            self.heads.extend([DenseGraphAttentionHead(input_dim, intermediate_dim)])

    def forward(self, nodes, edge_mat):
        head_outputs = []
        for head in self.heads:
            head_outputs.append(head(nodes, edge_mat))
        out = torch.cat(head_outputs, dim=-1)
        out = self.final_layer(out).softmax(dim=-1)
        return out

class DenseGraphAttentionNetwork(nn.Module):
    """
    Graph attention for Dense Graphs, where the edge list
    is given by an NxN array, where it is 1 if there is an
    edge from (i,j), 0 otherwise.
    """
    def __init__(self, depth, heads, input_dim, inner_dim):
        super(DenseGraphAttentionNetwork, self).__init__()
        self.layers = nn.ModuleList()
        if depth == 1:
            inner_dim = input_dim

        for layer_no in range(depth):
            if layer_no == 0:
                self.layers.extend([DenseGraphAttentionLayer(heads, input_dim, inner_dim, inner_dim)])
            elif layer_no == depth-1:
                self.layers.extend([DenseGraphAttentionLayer(heads, inner_dim, inner_dim, input_dim)])
            else:
                self.layers.extend([DenseGraphAttentionLayer(heads, inner_dim, inner_dim, inner_dim)])

    def forward(self, nodes, edge_mat):
        for layer in self.layers:
            nodes = layer(nodes, edge_mat)
        return nodes
