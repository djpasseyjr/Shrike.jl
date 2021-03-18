module RPTrees

    using AbstractTrees
    using StatsBase: median

    include("rptree.jl")
    include("nearest_neighbors.jl")

    export RPForest, traverse_to_leaves

end  # module RPTrees
