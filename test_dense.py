import torch
from fast_gat import DenseGraphAttentionNetwork

nodes = torch.tensor([[0.1, 0.2, 0.3],
                      [0.4, 0.5, 0.6],
                      [0.7, 0.8, 0.9],
                      [1.0, 1.1, 1.2]], dtype=torch.float)

edges = torch.tensor([[True, True, True, False],
                      [True, True, True, True],
                      [True, True, True, False],
                      [False,True, False,True]], dtype=torch.bool)

depth = 3
heads = 3
input_dim = 3
inner_dim = 2

net = DenseGraphAttentionNetwork(depth, heads, input_dim, inner_dim)

output = net(nodes, edges)
print(output)
