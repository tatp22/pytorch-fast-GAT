import torch
from fast_gat import GraphAttentionNetwork

nodes = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [1.0, 1.1, 1.2]], dtype= torch.float)
edges = {0: [1,2], 1: [0,2,3], 2: [0,1], 3: [1]}

depth = 3
heads = 3
input_dim = 3
inner_dim = 2

net = GraphAttentionNetwork(depth, heads, input_dim, inner_dim)

output = net(nodes, edges)
print(output)
