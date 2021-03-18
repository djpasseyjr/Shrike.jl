"""
    RPForest{T}

Multiple random projection tree type.

**Fields**
* `data::Array{T, 2}`: Contains all data points
* `npoints::Int`: Number of data points
* `ndims::Int`: Dimensionality of the data
* `depth::Int`: maximum depth of the tree. (Depth of 0 means only a root, depth of 1 means root has two children)
* `ntrees::Int`: Number of trees to make
* `random_vectors::AbstractArray`: The random projections used to make the tree
* `splits::Array{T, 2}`: The split values for each node in each tree stored in a 2D array
* `indexes::Array{Array{Int,1}, 2}`: 2D array of datapoint indexes at each leaf node in each tree.
Note that RP forest does not store indexes at non-leaf nodes.
"""

struct RPForest{T}
    data::Array{T, 2}
    npoints::Int
    ndims::Int
    depth::Int
    ntrees::Int
    random_vectors::AbstractArray
    splits::Array{T, 2}
    indexes::Array{Array{Int,1}, 2}
end

"""
    RPForest(data::Array{T, 2}, maxdepth::Int, ntrees::Int) -> ensemble

Constructor for ensemble of sparse random projection trees with voting. Follows the
implementation outlined in:

>**Fast Nearest Neighbor Search through Sparse Random Projections and Voting.**
Ville Hyvönen, Teemu Pitkänen, Sotirios Tasoulis, Elias Jääsaari, Risto Tuomainen,
Liang Wang, Jukka Ilmari Corander, Teemu Roos. Proceedings of the 2016 IEEE
Conference on Big Data (2016)
"""
function RPForest(data::Array{T, 2}, maxdepth::Int, ntrees::Int) where T
    # Need depth check
    ndims, npoints = size(data)
    sparse_vecs::Bool = ndims > 500
    random_vectors = random_projections(T, maxdepth*ntrees, ndims, sparse = sparse_vecs)
    nsplits = 2^maxdepth -1 # Number of splits in the tree (Same as number of non-leaf nodes)
    splits = zeros(T, ntrees, nsplits) # Array for storing hyperplane split values
    projections = random_vectors * data # projection[i, j] = the projection of point j onto the ith random vector
    nleafs = 2^maxdepth
    leaf_node_data_idxs = Array{Array{Int, 1}, 2}(undef, nleafs, ntrees) # Array for storing data indexes at the leaf

    for t in 1:ntrees
        prev_level = [collect(1:npoints)]
        next_level = Array{Array{Int, 1}, 2}(undef, 2, 1)
        next_split_idx = 1
        for d in 1:maxdepth
            proj_row = (t-1)*maxdepth + d # Current row of projection values
            for i in 1:length(prev_level)
                idxs = prev_level[i]
                # Compute median value of the projection of all current datapoints
                med::T = median(projections[proj_row, idxs]) # I think this requires a sort, so this process can be sped up slightly
                splits[t, next_split_idx] = med
                next_split_idx += 1
                # Separate data points across hyperplane using the median
                mask = projections[proj_row, idxs] .< med
                # Send points with the projection less than the median to the left child
                next_level[2*i - 1] = idxs[mask]
                # Send points with the projection greater than the median to the right child
                next_level[2*i] = idxs[.!mask]

            end
            prev_level = next_level
            next_level = Array{Array{Int, 1}, 2}(undef, 2^(d+1), 1)
        end
        leaf_node_data_idxs[:, t] = prev_level
    end

    return RPForest(data, npoints, ndims, maxdepth, ntrees, random_vectors, splits, leaf_node_data_idxs)
end

function Base.show(io::IO, rpf::RPForest)
    descr = "RPForest: \n    $(rpf.ntrees) trees \n    $(rpf.npoints) datapoints \n    Depth $(rpf.depth)"
    print(io, descr)
end

function random_projections(T::Type, nvecs::Int, ndims::Int; sparse::Bool=true)
    # Could use static arrays here
    if sparse
        println("Not implemented")
    else
        R = randn(T, nvecs, ndims)
        R ./= sum(R.^2, dims=2) .^ 0.5
    end
    return R::Array{T, 2}
end

function medianperm(x::Array{T, 1}) where T
    sortidx = sortperm(x)
    mi = isodd(n) ? n ÷ 2 + 1 : n ÷ 2
    median = isodd(n) ? dimvals[mi] : (dimvals[mi] + dimvals[mi+1]) / 2
    lidxs = node.indexes[sortidx][1:mi]
    ridxs = node.indexes[sortidx][mi+1:end]
end
