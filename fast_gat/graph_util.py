import torch
import torch.nn as nn

class GraphUtils:
    """
    This class provides an easy way to convert a
    graph's edges from an adjacency list data store to an
    adjacency matrix and back again.
    """
    @staticmethod
    def list_to_matrix(num_nodes, edge_list):
        mat = torch.eye(num_nodes, dtype=torch.bool)
        for start, dests in edge_list.items():
            for dest in dests:
                mat[start, dest] = True
        return mat

    @staticmethod
    def matrix_to_list(edge_matrix):
        l = dict()
        for i, neighbors in enumerate(edge_matrix):
            l[i] = set(map(lambda tensor: tensor.item(), torch.nonzero(neighbors).flatten()))
            l[i].discard(i)
        return l
