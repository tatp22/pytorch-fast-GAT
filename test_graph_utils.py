import torch
from fast_gat import GraphUtils

edges = {0: {1,2}, 1: {0,2,3}, 2: {0,1}, 3: {1}}
num_nodes = 4
out = GraphUtils.matrix_to_list(GraphUtils.list_to_matrix(num_nodes, edges))
assert out == edges, "The graph utils function failed because {} != {}".format(out, edges)
print("method works!")
