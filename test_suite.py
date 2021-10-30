import torch
from fast_gat import DenseGraphAttentionNetwork, GraphAttentionNetwork, GraphUtils

num_nodes = 4
nodes = torch.tensor([[0.1, 0.2, 0.3],
                      [0.4, 0.5, 0.6],
                      [0.7, 0.8, 0.9],
                      [1.0, 1.1, 1.2]], dtype=torch.float)

sparse_edges = {0: {1,2}, 1: {0,2,3}, 2: {0,1}, 3: {1}}
full_edges = torch.tensor([[True, True, True, False],
                           [True, True, True, True],
                           [True, True, True, False],
                           [False,True, False,True]], dtype=torch.bool)

depth = 3
heads = 3
input_dim = 3
inner_dim = 2

def test_dense():
    net = DenseGraphAttentionNetwork(depth, heads, input_dim, inner_dim)
    output = net(nodes, full_edges)

def test_sparse():
    net = GraphAttentionNetwork(depth, heads, input_dim, inner_dim)
    output = net(nodes, sparse_edges)

def test_sparse_graph_utils():
    sparse_test = GraphUtils.matrix_to_list(GraphUtils.list_to_matrix(num_nodes, sparse_edges))
    assert sparse_test == sparse_edges, "The graph utils function failed for the sparse case!"

def test_dense_graph_utils():
    dense_test = GraphUtils.list_to_matrix(num_nodes, GraphUtils.matrix_to_list(full_edges))
    assert torch.equal(dense_test, full_edges), "The graph utils function failed for the dense case!"
