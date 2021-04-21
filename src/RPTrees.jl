module RPTrees

    using AbstractTrees
    using DataStructures
    using Distances
    using LightGraphs
    using LinearAlgebra
    using StatsBase: median
    using SparseArrays

    include("rptree.jl")
    include("nearest_neighbors.jl")

    export RPForest, traverse_to_leaves, approx_knn

end  # module RPTrees
