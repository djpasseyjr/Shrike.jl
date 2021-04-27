# RPTrees.jl

![Random Projection Splits](https://github.com/djpasseyjr/RPTrees.jl/raw/main/docs/src/images/rppartition.png)

`RPTrees` is a Julia package for building ensembles of random projection trees. Random projection trees are a generalization of KD-Trees and are used to quickly approximate nearest neighbors or build k-nearest-neighbor graphs. They [conform to low dimensionality](https://cseweb.ucsd.edu/~dasgupta/papers/rptree-stoc.pdf) that is often present in high dimensional data.

The implementation here is based on the [MRPT algorithm](https://helda.helsinki.fi//bitstream/handle/10138/301147/Hyvonen_Pitkanen_2016_Fast_Nearest.pdf?sequence=1). This package also includes optimizations for knn-graph creation and has built-in support for multithreading.

## Installation

To install just type

```jl
] add https://github.com/djpasseyjr/RPTrees.jl
```

in the REPL or 

```jl 
using Pkg
Pkg.add(path="https://github.com/djpasseyjr/RPTrees.jl")
```

## Build an Index

To build an ensemble of random projection trees use the `RPForest` type.

```jl
X = rand(100, 10000)
rpf = RPForest(X; depth=6, ntrees=5)
```
The type accepts a matrix of data, `X` where each column represents a datapoint. 

1. `depth` describes the number of times each random projection tree will split the data. Leaf nodes in the tree contain `npoints / 2^depth` data points. Increasing `depth` increases speed but decreases accuracy.
2. `ntrees` controls the number of trees in the ensemble. More trees means more accuracy but more memory.

To query the index use
```jl
k = 10
q = X[:, 1]
ann = approx_knn(rpf, q, k; vote_cutoff=2)
```

1. The `vote_cutoff` parameter signifies how many "votes" a point needs in order to be included in a linear search. Each tree "votes" for the points a leaf node, so if there aren't many point in the leaves and there aren't many trees, the odds of a point receiving more than one vote is low.  Increasing `vote_cutoff` speeds up the algorithm but may reduce accuracy. When `depth` is large and `ntrees` is less than 5, it is reccomended to set `vote_cutoff = 1`. 

## KNN-Graphs

This package was designed specifically to generate k-nearest-neighbor graphs and has specialized functions for this purpose. It uses neighbor of neighbor exploration (outlined [here](https://arxiv.org/pdf/1602.00370.pdf)) to efficiently improve the accuracy of a knn-graph.

Nearest neighbor graphs are used to give a sparse topology to large datasets. Their structure can be used to [project the data](https://arxiv.org/pdf/1602.00370.pdf) onto a lower dimensional manifold, to cluster datapoints with community detection algorithms or to preform other analyses.

To generate nearest neighbor graphs:

```jl
X = rand(100, 10000)
rpf = RPForest(X; depth=6, ntrees=5)
k = 10
g = knngraph(rpf, k; vote_cutoff=1, ne_iters=1, gtype=SimpleDiGraph)
```
1. The `vote_cutoff` parameter signifies how many "votes" a point needs in order to be included in a linear search.
2. `ne_iters` controlls how many iterations of neighbor exploration the algorithm will undergo. Successive iterations are increasingly fast. It is reccomened to use more iterations of neighbor exploration when the number of trees is small and less when many trees are used.
3. The `gtype` parameter allows the user to specify a `LightGraphs.jl` graph type to return. `gtype=identity` returns a sparse adjacency matrix.

If an array of nearest neighbor indices is preferred,

```jl
nn = allknn(rpf, k; vote_cutoff=1, ne_iters=0)
```

can be used to generate an `rpf.npoints`x`k` array of integer indexes where `nn[i, :]` corresponds to the nearest neighbors of `X[:, i]`. The keyword arguments work as outlined above.

## Threading

`RPTrees` has built in support for multithreading. To allocate multiple threads, start `julia` with the `--threads` flag:

```console
user@sys:~$ julia --threads 4
```

To see this at work, consider a small scale example:
```console
user@sys:~$ cmd="using RPTrees; rpf=RPForest(rand(100, 10000)); @time knngraph(rpf, 10, ne_iters=1)"
user@sys:~$ julia -e "$cmd"
29.422300 seconds (140.91 M allocations: 8.520 GiB, 5.86% gc time)
user@sys:~$ julia -e --threads 4 "$cmd"
15.212044 seconds (141.42 M allocations: 8.840 GiB, 5.98% gc time)
```
(This assumes that `RPTrees` is installed.)

## Benchmarks

![Benchmark Plot](images/benchplot.png)

## Function Documentation

```@docs
RPForest(data::AbstractArray{T, 2}, depth::Int, ntrees::Int) where T
```

```@docs
approx_knn(rpf::RPForest{T}, q::AbstractArray{T, 2}, k::Int; vote_cutoff=1) where T
```

```@docs
knngraph(rpf::RPForest{T}, k::Int; vote_cutoff::Int=1, ne_iters::Int=0, gtype::G=SimpleDiGraph) where {T, G}
```

```@docs
explore(i::Int, data::AbstractArray{T}, ann::Array{NeighborExplorer{T}, 1}) where T
```

```@docs
allknn(rpf::RPForest{T}, k::Int; vote_cutoff::Int=1, ne_iters::Int=0) where T
```


```@docs
traverse_to_leaves(rpf::RPForest{T}, x::Array{T, 2}) where T
```
