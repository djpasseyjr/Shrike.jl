struct ShrikeIndex{T <: AbstractFloat}
    data::Array{T, 2}
    npoints::Int
    ndims::Int
    depth::Int
    ntrees::Int
    random_vectors::Array{T, 2}
    splits::Array{T, 2}
    indexes::Array{Array{Int,1}, 2}
end

"""
    ShrikeIndex(data::Array{T, 2}, depth::Int, ntrees::Int) where T -> ensemble

Constructor for ensemble of sparse random projection trees with voting. Returns `ShrikeIndex` type.
(An ensemble of multiple random projection trees.)

** Type Fields**
* `data::Array{T, 2}`: Contains all data points
* `npoints::Int`: Number of data points
* `ndims::Int`: Dimensionality of the data
* `depth::Int`: maximum depth of the tree. (Depth of 0 means only a root, depth of 1 means root has two children)
* `ntrees::Int`: Number of trees to make
* `random_vectors::AbstractArray`: The random projections used to make the tree
* `splits::Array{T, 2}`: The split values for each node in each tree stored in a 2D array
* `indexes::Array{Array{Int,1}, 2}`: 2D array of datapoint indexes at each leaf node in each tree.
Note that RP forest does not store indexes at non-leaf nodes.

Follows the implementation outlined in:

>**Fast Nearest Neighbor Search through Sparse Random Projections and Voting.**
>Ville Hyvönen, Teemu Pitkänen, Sotirios Tasoulis, Elias Jääsaari, Risto Tuomainen,
>Liang Wang, Jukka Ilmari Corander, Teemu Roos. Proceedings of the 2016 IEEE
>Conference on Big Data (2016)

with some modifications.
"""
function ShrikeIndex(data::AbstractArray{T, 2}, depth::Int, ntrees::Int) where T
    ndims, npoints = size(data)
    maxdepth = floor(Int, log(2.0, npoints) - log(2.0, 1))
    if depth > maxdepth
        depth = maxdepth
    end
    sparse_vecs::Bool = ndims > 500
    random_vectors = random_projections(T, depth*ntrees, ndims, sparse = sparse_vecs)
    nsplits = 2^depth -1 # Number of splits in the tree (Same as number of non-leaf nodes)
    splits = zeros(T, ntrees, nsplits) # Array for storing hyperplane split values
    projections = transpose(data) * random_vectors  # projection[i, j] = the projection of point i onto the jth random vector
    nleafs = 2^depth
    leaf_node_data_idxs = Array{Array{Int, 1}, 2}(undef, nleafs, ntrees) # Array for storing data indexes at the leaf

    for t in 1:ntrees
        prev_level = [collect(1:npoints)]
        next_level = Array{Array{Int, 1}, 2}(undef, 2, 1)
        next_split_idx = 1
        for d in 1:depth
            proj_row = (t-1)*depth + d # Current row of projection values
            for i in 1:length(prev_level)
                idxs = prev_level[i]
                # Compute median value of the projection of all current datapoints
                p = reshape(projections[idxs, proj_row], :)
                med::T = median(p) # I think this requires a sort, so this process can be sped up slightly
                splits[t, next_split_idx] = med
                next_split_idx += 1
                # Separate data points across hyperplane using the median
                mask = p .< med
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

    return ShrikeIndex(data, npoints, ndims, depth, ntrees, random_vectors, splits, leaf_node_data_idxs)
end

ShrikeIndex(data::AbstractArray{T, 2}; depth::Int=4, ntrees::Int=5) where T = ShrikeIndex(data, depth, ntrees)

"""
    ShrikeIndex(data::AbstractArray{T, 2}, max)k; depth::Union{Int, Float64}=Inf, ntrees::Int=5) -> shi

Keyword argument version of the constructor that includes intended number of nearest neighbors.

If the default `depth` is used, the constructor sets the tree depth as deep as
possible given `max_k`. This way, the accuracy/memory tradeoff is determined directly by `ntrees` and the
desired `vote_cutoff` (`vote_cutoff` is a parameter passed to `ann` or `knngraph`).

If an argument is passed for `depth`, constructor attempts to use the supplied `depth`, but guarentees
that the depth of the tree is shallow enough to ensure that each leaf has at least k points.
(Without this check, the index may return less than k neighbors when queried.)

**Parameters**

1. `data`: A (dxn) array. Each column is a datapoint with dimension `d`.
2. `max_k`: The maximum number of neighbors that will be queried. If intend
to use the `ShrikeIndex` to approximate at most 10 nearest neigbors of a point,
set `max_k = 10`. This argument is used to infer the deepest tree depth possible
so as to maximize speed,

**Keyword Arguments**

3. `ntrees`: The number of trees in the index. More trees means more accuracy,
more memory and less speed. Use this to tune the speed/accuracy tradeoff.
4. `depth`: The number of splits in the tree. Depth of 0 means only a root,
depth of 1 means root has two children, etc..
"""
function ShrikeIndex(data::AbstractArray{T, 2}, maxk::Int; depth::Union{Int, Float64}=Inf, ntrees::Int=10) where T
    m, n = size(data)
    maxdepth = floor(Int, log(2, n) - log(2, maxk))
    if depth > maxdepth
        safedepth = maxdepth
    else
        safedepth = depth
    end
    return ShrikeIndex(data, safedepth, ntrees)
end


function Base.show(io::IO, shi::ShrikeIndex)
    descr = "ShrikeIndex: \n    $(shi.ntrees) trees \n    $(shi.npoints) datapoints \n    Depth $(shi.depth)"
    print(io, descr)
end

function random_projections(T::Type, nvecs::Int, ndims::Int; sparse::Bool=true)
    # Sparse arrays are not implemented because sparse x dense matmul
    # was slower than dense x dense matmul when benchmarked.
    # Additionally, I'm not sure how to use sparse arrays and still
    # be type stable.
    # a::Float64 = 1. / √ ndims
    # if sparse
    #     R = sprand(ndims, nvecs, a)
    # else
        R = randn(T, ndims, nvecs)
        R ./= sum(R.^2, dims=2) .^ 0.5
    # end
    return R::Array{T, 2}
end
