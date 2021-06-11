# Shrike.jl
[![Build Status](https://github.com/djpasseyjr/Shrike.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/djpasseyjr/Shrike.jl/actions/workflows/ci.yml/badge.svg)
[![codecov](https://codecov.io/gh/djpasseyjr/Shrike.jl/branch/main/graph/badge.svg?token=S7PNXQOLQK)](https://codecov.io/gh/djpasseyjr/Shrike.jl)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://djpasseyjr.github.io/Shrike.jl/dev)

![Random Projection Splits](https://github.com/djpasseyjr/Shrike.jl/raw/main/docs/src/images/rppartition.png)

`Shrike` is a Julia package for building ensembles of random projection trees. Random projection trees are a generalization of KD-Trees and are used to quickly approximate nearest neighbors or build k-nearest-neighbor graphs. They [conform to low dimensionality](https://cseweb.ucsd.edu/~dasgupta/papers/rptree-stoc.pdf) that is often present in high dimensional data.

The implementation here is based on the [MRPT algorithm](https://helda.helsinki.fi//bitstream/handle/10138/301147/Hyvonen_Pitkanen_2016_Fast_Nearest.pdf?sequence=1). This package also includes optimizations for knn-graph creation and has built-in support for multithreading.

## Installation

To install just type

```jl
] add https://github.com/djpasseyjr/Shrike.jl
```

in the REPL or

```jl
using Pkg
Pkg.add(path="https://github.com/djpasseyjr/Shrike.jl")
```

## Build an Index

To build an ensemble of random projection trees use the `ShrikeIndex` type.

```jl
using Shrike
X = rand(100, 10000)
shi = ShrikeIndex(X; maxdepth=6, ntrees=5)
```
The type accepts a matrix of data, `X` where each column represents a datapoint.

1. `maxdepth` describes the number of times each random projection tree will split the data. Leaf nodes in the tree contain `npoints / 2^maxdepth` data points. Increasing `maxdepth` increases speed but decreases accuracy.
2. `ntrees` controls the number of trees in the ensemble. More trees means more accuracy but more memory.

To query the index for approximte nearest neighbors use
```jl
k = 10
q = X[:, 1]
ann = ann(shi, q, k; vote_cutoff=2)
```

1. The `vote_cutoff` parameter signifies how many "votes" a point needs in order to be included in a linear search. Each tree "votes" for the points a leaf node, so if there aren't many point in the leaves and there aren't many trees, the odds of a point receiving more than one vote is low.  Increasing `vote_cutoff` speeds up the algorithm but may reduce accuracy. When `maxdepth` is large and `ntrees` is less than 5, it is reccomended to set `vote_cutoff = 1`.

## KNN-Graphs

This package was designed specifically to generate k-nearest-neighbor graphs and has specialized functions for this purpose. It uses neighbor of neighbor exploration (outlined [here](https://arxiv.org/pdf/1602.00370.pdf)) to efficiently improve the accuracy of a knn-graph.

Nearest neighbor graphs are used to give a sparse topology to large datasets. Their structure can be used to [project the data](https://arxiv.org/pdf/1602.00370.pdf) onto a lower dimensional manifold, to cluster datapoints with community detection algorithms or to preform other analyses.

To generate nearest neighbor graphs:

```jl
using Shrike
X = rand(100, 10000)
shi = ShrikeIndex(X; maxdepth=6, ntrees=5)
k = 10
g = knngraph(shi, k; vote_cutoff=1, ne_iters=1, gtype=SimpleDiGraph)
```
1. The `vote_cutoff` parameter signifies how many "votes" a point needs in order to be included in a linear search.
2. `ne_iters` controlls how many iterations of neighbor exploration the algorithm will undergo. Successive iterations are increasingly fast. It is reccomened to use more iterations of neighbor exploration when the number of trees is small and less when many trees are used.
3. The `gtype` parameter allows the user to specify a `LightGraphs.jl` graph type to return. `gtype=identity` returns a sparse adjacency matrix.

If an array of nearest neighbor indices is preferred,

```jl
nn = allknn(shi, k; vote_cutoff=1, ne_iters=0)
```

can be used to generate an `shi.npoints`x`k` array of integer indexes where `nn[i, :]` corresponds to the nearest neighbors of `X[:, i]`. The keyword arguments work as outlined above.

## Threading

`Shrike` has built in support for multithreading. To allocate multiple threads, start `julia` with the `--threads` flag:

```console
user@sys:~$ julia --threads 4
```

To see this at work, consider a small scale example:
```console
user@sys:~$ cmd="using Shrike; shi=ShrikeIndex(rand(100, 10000)); @time knngraph(shi, 10, ne_iters=1)"
user@sys:~$ julia -e "$cmd"
  12.373127 seconds (8.66 M allocations: 4.510 GiB, 6.85% gc time, 18.88% compilation time)
user@sys:~$ julia  --threads 4 -e "$cmd"
  6.306410 seconds (8.67 M allocations: 4.498 GiB, 13.12% gc time, 31.64% compilation time)
```
(This assumes that `Shrike` is installed.)

## Benchmark

This package was compared to the original [`mrpt`](https://github.com/vioshyvo/mrpt) C++ implementation (on which this algorithm was based), [`annoy`](https://github.com/spotify/annoy), a popular package for approximate nearest neighbors, and [`NearestNeighbors.jl`](https://github.com/KristofferC/NearestNeighbors.jl), a Julia package for nearest neighbor search. The benchmarks were written in the spirit of [`ann-benchmarks`](https://github.com/erikbern/ann-benchmarks), a repository for comparing different approximate nearest neighbor algorithms. The datasets used for the benchmark were taken directly from `ann-benchmarks`. The following are links to the HDF5 files in question: [FashionMNIST](http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5), [SIFT](http://ann-benchmarks.com/sift-128-euclidean.hdf5), [MNIST](http://ann-benchmarks.com/mnist-784-euclidean.hdf5) and [GIST](http://ann-benchmarks.com/gist-960-euclidean.hdf5). The benchmarks below were run on a compute cluster, restricting all algorithms to a single thread.

![FashionMNIST Speed Comparison](https://github.com/djpasseyjr/Shrike.jl/raw/main/docs/src/images/fashionmnist_bm.png)

In this plot, up and to the right is better. (Faster queries, better recall). Each point represents a parameter combination. For a full documentation of parameters run and timing methods consult the original scripts located in the `benchmark/` directory.

This plot illustrates how for this dataset, on most parameter combinations, `Shrike` has better preformance. Compared to SIFT, below, where some parameter combinations are not as strong. We speculate that this has to do with the high dimensionality of points in FashionMNIST (d=784), compared to the lower dimensionality of SIFT (d=128).

![SIFT Speed Comparison](https://github.com/djpasseyjr/Shrike.jl/raw/main/docs/src/images/sift_bm.png)

It is important to note that `NearestNeighbors.jl` was designed to return the *exact* k-nearest-neighbors as quickly as possible, and does not approximate, hence the high accuracy and lower speed.

The takeaway here is that `Shrike` is fast! It is possibly a little faster than the original C++ implementation. Go Julia! We should note, that `Shrike` was *not* benchmarked against state of the art algorithms for approximate nearest neighbor search. These algorithms are faster than `annoy` and `mrpt`, but unfortunately, the developers of `Shrike` aren't familiar with these algorithms.
