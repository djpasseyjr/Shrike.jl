# RPTrees

Create ensembles of random projection trees in Julia.

# Functions

```@docs
RPForest(data::AbstractArray{T, 2}, maxdepth::Int, ntrees::Int) where T
```

```@docs
traverse_to_leaves(rpf::RPForest{T}, x::Array{T, 2}) where T
```
