# RPTrees

Create ensembles of random projection trees in Julia.

# Functions

```@docs
RPForest(data::AbstractArray{T, 2}, maxdepth::Int, ntrees::Int) where T
```

```@docs
knngraph(rpf::RPForest{T}, k::Int; vote_cutoff::Int=1, ne_iters::Int=0, gtype::G=SimpleDiGraph) where {T, G}
```

```@docs
approx_knn(rpf::RPForest{T}, q::AbstractArray{T, 2}, k::Int; vote_cutoff=1) where T
```

```@docs
allknn(rpf::RPForest{T}, k::Int; vote_cutoff::Int=1, ne_iters::Int=0) where T
```

```@docs
traverse_to_leaves(rpf::RPForest{T}, x::Array{T, 2}) where T
```
