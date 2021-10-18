# Pytorch Fast GAT Implementation

This is my implementation of an old paper, [Graph Attention Networks](https://arxiv.org/pdf/1710.10903.pdf).
However, instead of a standard impementation, this one introduces several techniques to speed up the process,
which are found below.

## Installation

```
pip install <NAME-TBA>
```

Alternatively,

```
git clone https://github.com/tatp22/pytorch-fast-GAT.git
cd fast-gat
```

## What makes this repo faster?

What is great about this paper is that, besides its state of the art performance on a number of benchmarks,
is that it could be applied to any graph, regardless of its structure. However, this algorithm has a runtime
that depends on the number of edges, and when the graph is dense, this means that it can run in `nodes^2` time.

Most sparsifying techniques for graphs rely on somehow decreasing the number of edges. However, I will try out
a different method: Reducing the number of nodes in the interior representation. This will be done similarly to how
the [Linformer](https://arxiv.org/pdf/2006.04768.pdf) decreases the memory requirement of the internal matrices, which
is by adding a parameterized matrix to the input that transforms the input. A challenge here is that since this is a graph,
not all nodes will connect to all other nodes. This is why I plan to introduce a masking technique, which is explained further
down.
(not yet implemented)

Note: This idea has not been tested. I do not know what its performance will be on real life applications,
and it may or may not provide accurate results.

## Code Example

TODO

## Downsampling method

TODO
