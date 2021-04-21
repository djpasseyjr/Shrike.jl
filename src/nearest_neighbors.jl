"""
    traverse_to_leaves(rpf::RPForest{T}, x::Array{T, 2}) where T -> leaf_idxs

Route data point `x` down to a leaf node each tree and return and array of
indexes of the data stored in each corresponding leaf node
"""
function traverse_to_leaves(rpf::RPForest{T}, x::AbstractArray{T, 2}) where T
    #TODO Dispatch on shape of x?
    # Compute all needed projections
    proj::Array{T, 1} = reshape(transpose(x) * rpf.random_vectors, :)
    # Initial node indexes (Root index for each tree)
    node_idxs = ones(Int, rpf.ntrees)
    # Location of projection value for the current depth (Starts at
    # [1, rpf.depth, 2*rpf.depth, ... rpf.depth*rpf.ntrees].)
    projection_idxs::Array{Int, 1} = collect(1:rpf.depth:rpf.depth*rpf.ntrees);
    # For each level of the trees
    for d in 1:rpf.depth
        # Determine if the projection is on the left or right of the split
        mask::Array{Bool, 1} = proj[projection_idxs] .>=  _get_splits(rpf, node_idxs)
        # Move node indexes to the right child
        node_idxs .*= 2
        # Change to left child if the projection was less than the split value
        node_idxs[mask] .+= 1
        # Increase depth of projection locations by one
        projection_idxs .+= 1
    end
    # Subtract the number of non leaf nodes (Because we don't store indexes
    # at non leaf nodes)
    node_idxs .-= 2^rpf.depth - 1
    leaf_idxs = map(i -> rpf.indexes[node_idxs[i], i], 1:rpf.ntrees)
    return leaf_idxs
end

"""
    _get_splits(rpf::RPForest{T}, node_idx::Array{Int, 1}) where T -> splits

Given an array of node indexes of length `rpf.ntrees` corresponding to the
current index in each tree, return the split values of each node.
"""
_get_splits(rpf::RPForest{T}, node_idx::Array{Int, 1}) where T = map(i -> rpf.splits[i, node_idx[i]], 1:rpf.ntrees)

mutable struct NeighborExplorer{T}
    k::Int
    heap::BinaryMaxHeap{Tuple{T, Int}}
    npoints::Int
    idx_set::Set{Int}
end

"""
    NeighborExplorer(idx::Int, k::Int) where T -> ne

Returns a NeighborExlorer type, a modified max heap for exploring neighbors of 
neighbors during the creation of a k-nearest-neighbors graph

**Parameters:**
    
* `idx` (Int): The current datapoint or node. (Should not be included in the k nearest neighbors)
* `k` (Int): Maximum number of neighbors to store

"""
function NeighborExplorer{T}(idx::Int, k::Int) where T
    heap = BinaryMaxHeap{Tuple{T, Int}}()
    sizehint!(heap, k+2)
    idx_set = Set{Int}([idx])
    return NeighborExplorer{T}(k, heap, 0,  idx_set)
end
    
"""
    Base.push!(ne::NeighborSearcher{T}, idx::Int, dist::T) where T
   
Store the datapoint or node with index `idx` with distances `dist` from
the current datapoint in `ne`. 
"""

function Base.push!(ne::NeighborExplorer{T}, idx::Int, dist::T) where T
    if idx in ne.idx_set
        return
    end
    ne.npoints += 1
    push!(ne.idx_set, idx)
    push!(ne.heap, (dist, idx))
    if ne.npoints == ne.k + 1
        pop!(ne)
    end
end

"""
    Base.pop!(ne::NeighborExplorer{T}) where T -> idx, dist

Remove the node or datapoint with the largest distance from the neighbor
explorer.
"""
function Base.pop!(ne::NeighborExplorer{T}) where T
    # Keep seen indexes in the index set so we don't consider them again
    maxdist = pop!(ne.heap)
    ne.npoints -= 1
    return maxdist
end

"""
    get_dists(ne::NeighborExplorer) -> neighbor_idxs
    
Return the current distances of `k` nearest neighbors.
"""
get_dists(ne::NeighborExplorer) = [dist for (dist, idx) in ne.heap.valtree]

"""
    get_idxs(ne::NeighborExplorer) -> neighbor_idxs
    
Return the current indicies to `k` nearest neighbors.
"""
get_idxs(ne::NeighborExplorer) = [idx for (dist, idx) in ne.heap.valtree]


function Base.show(io::IO, ne::NeighborExplorer)
    descr = "NeighborExplorer: \n    Currently holds $(ne.npoints)/$(ne.k) points \n    $(length(ne.idx_set)) points already explored "
    print(io, descr)
end

""" Helper function for RP-Tree voting """
vote!(v::Array{Int, 1}, idx::Array{Int, 1}) = v[idx] .+= 1

"""
    candidate_idxs(rpf::RPForest{T}, q::Array{T, 2}, k::Int; vote_cutoff=1) where T -> cand_idx

Each RP-Tree in the `RPForest` "votes" for nearest neighbor candidates. All points that receive 
more than `vote_cutoff` votes are returned. By default `vote_cutoff=1` and this returns 
the set union of all RP-Tree leaf nodes in the ensemble. This is the standard RP-Tree algorithm.
"""
function candidate_idxs(rpf::RPForest{T}, q::AbstractArray{T, 2}, k::Int; vote_cutoff=1) where T
    # Get indexes in leaf nodes corresponding to `q`
    leaf_idxs = traverse_to_leaves(rpf, q)
    # Empty array of votes
    votes = zeros(Int, rpf.npoints)
    # Each tree votes on candiate nearest neighbors
    map(idxs -> vote!(votes, idxs), leaf_idxs)
    # Find candiate points with enough votes
    vote_mask = votes .>= vote_cutoff
    cand_idx::Array{Int, 1} = findall(vote_mask)
    return cand_idx
end

"""
    approx_knn(rpf::RPForest{T}, q::Array{T, 2}, k::Int; vote_cutoff=1) where T -> knn_idx

For a query point `q`, find the approximate `k` nearest neighbors from the data stored in the the
RPForest. The `vote_cutoff` parameter signifies how many "votes" a point needs in order to be included in a linear search. Increasing `vote_cutoff` speeds up the algorithm but may reduce accuracy.

"""
function approx_knn(rpf::RPForest{T}, q::AbstractArray{T, 2}, k::Int; vote_cutoff=1) where T
    metric = Euclidean()
    cand_idx = candidate_idxs(rpf, q, k, vote_cutoff=vote_cutoff)
    ncand::Int = length(cand_idx)
    # Linear search on candidates
    cand_dist = zeros(T, ncand)
    @inbounds for i in 1:ncand
        x = @view rpf.data[:, i]
        cand_dist[i] = metric(x, q)
    end
    sp = sortperm(cand_dist)
    knn_idx = [cand_idx[sp[i]] for i=1:k]
    return knn_idx
end


# """
#     approx_knn(rpf::RPForest{T}, q::Array{T, 2}, k::Int; vote_cutoff=1) where T -> knn_ne

# Helper function for creating a knn-graph. Instead of returning indexes, returns a 
# `NeighborExplorer` containing the approximate nearest neighbors. 

# The `vote_cutoff` parameter signifies how many "votes" a point needs in order to be included in a linear search. Increasing `vote_cutoff` speeds up the algorithm but may reduce accuracy.
# """
# function approx_knn(rpf::RPForest{T}, i::Int, k::Int; vote_cutoff=1) where T
#     metric = Euclidean()
#     q = @view rpf.data[:, i:i]
#     cand_idx = candidate_idxs(rpf, q, k, vote_cutoff=vote_cutoff)
#     ncand::Int = length(cand_idx)
#     # Linear search on candidates
#     ne = NeighborExplorer{T}(i, k)
#     @inbounds for (i, c) in enumerate(cand_idx)
#         x = @view rpf.data[:, c]
#         push!(ne, c, metric(x, q))
#     end
#     return ne
# end

"""
    vote!(vs::Array{Dict{Int,Int}, 1}, idxs::Array{Int, 1})

Helper function for creating a knn-graph from a RPForest
"""
function vote!(vs::Array{Dict{Int,Int}, 1}, idxs::Array{Int, 1})
    for i in idxs
        for j in idxs
            v = get(vs[i], j, 0)
            vs[i][j] = v + 1
            u = get(vs[j], i, 0)
            vs[j][i] = u + 1
        end
    end
end

"""
    _allknn(rpf::RPForest{T}, k::Int; vote_cutoff=1) where T -> knn

Returns a neighbor explorer for each point to allow for neighbors of neighbors exploration
and the creation of a knn-graph. 

The `vote_cutoff` parameter signifies how many "votes" a point needs in order to be included in a linear search. Increasing `vote_cutoff` speeds up the algorithm but may reduce accuracy.
"""
function _allknn(rpf::RPForest{T}, k::Int; vote_cutoff=1) where T
    metric = Euclidean()
    votes = [Dict{Int,Int}() for i in 1:rpf.npoints]
    knn = [NeighborExplorer{T}(i, k) for i in 1:rpf.npoints]
    nleafs = 2^rpf.depth
    for i in 1:rpf.ntrees
        for j in 1:nleafs
            vote!(votes, rpf.indexes[j, i])
        end
    end
    
    for i in 1:rpf.npoints
        nodes = collect(keys(votes[i]))
        distances = zeros(T, length(nodes))
        x = @view rpf.data[:, i]
        for node in nodes
            if votes[i][node] > vote_cutoff
                y = @view rpf.data[:, node]
                # Should we sort before adding to heap?
                push!(knn[i], node, metric(x,y))
            end
        end
    end
    return knn
end

"""
    explore!(data::Array{T, 2}, ann::Array{NeighborExplorer{T}, 1}) where T

Explore neighbors of neighbors to improve accuracy of the approximate 
k-nearest-neigbors stored in `knn`.

**Parameters**

1. `data` is the array of data. Each column is a data point

2. `ann` is an array of `NeighborExplorer`s where `ann[i]` contains 
the current best approximation to the k-nearest-neigbors of point `i`.

"""
function explore!(data::Array{T, 2}, ann::Array{NeighborExplorer{T}, 1}) where T
    m,n = size(data)
    metric = Euclidean()
    @inbounds for i in 1:n
        x = @view data[:, i:i]
        for j in get_idxs(ann[i])
            for l in get_idxs(ann[j])
                y = @view data[:, l:l]
                dist = metric(x, y)
                push!(ann[i], l, dist)
            end
        end
    end
end

"""
    allknn(rpf::RPForest{T}, k::Int; vote_cutoff::Int=1, ne_iters::Int=0) where T -> approxnn

Returns a `rpf.npoints` by `k` array of approximate nearest neighbor indexes.
That is, `approxnn[i,:]` contains the indexes of the k nearest neighbors of
`rpf.data[:, i]`.

The `ne_iters` assigns the number of iterations of neighbor exploration to use. 
Neighbor exploration is an inexpensive way to increase accuracy.

The `vote_cutoff` parameter signifies how many "votes" a point needs in order to be included 
in a linear search. Increasing `vote_cutoff` speeds up the algorithm but may reduce accuracy.
"""
function allknn(rpf::RPForest{T}, k::Int; vote_cutoff::Int=1, ne_iters::Int=0) where T
    # Search tree via neighbor explorers
    ann = _allknn(rpf, k, vote_cutoff=vote_cutoff)
    # Neighbor exploration
    for i in 1:ne_iters
        explore!(rpf.data, ann)
    end
    # Load neighbor explorers into a matrix
    approxnn = zeros(Int, rpf.npoints, k)
    for (i, ne) in enumerate(ann)
        @inbounds approxnn[i, :] = get_idxs(ne) 
    end
    return approxnn
end

"""
    knngraph(rpf::RPForest{T}, k::Int, vote_cutoff; vote_cutoff::Int=1, ne_iters::Int=0, gtype::G) where {T, G} -> g

Returns a graph with `rpf.npoints` node and `k * rpf.npoints` edges datapoints conneceted to nearest neighbors

**Parameters**

1. `rpf`: random forest of the desired data
2. `k`: the desired number of nearest neighbors
3. `vote_cutoff`: signifies how many "votes" a point needs in order to be included 
in a linear search through leaf nodes. Increasing `vote_cutoff` speeds up the algorithm but may reduce accuracy. Defaults to 1
4. `ne_iters`: assigns the number of iterations of neighbor exploration to use. Defaults to zero.
Neighbor exploration is a way to increse knn-graph accuracy.
5. `gtype` is the type of graph to construct. Defaults to `SimpleDiGraph`

"""
function knngraph(rpf::RPForest{T}, k::Int; vote_cutoff::Int=1, ne_iters::Int=0, gtype::G=SimpleDiGraph) where {T, G}
    # Search tree via neighbor explorers
    ann = _allknn(rpf, k, vote_cutoff=vote_cutoff)
    # Neighbor exploration
    for i in 1:ne_iters
        explore!(rpf.data, ann)
    end
    # Load neighbor explorers into a sparse adj matrix
    A = spzeros(Int, rpf.npoints, rpf.npoints)
    @inbounds for j in 1:rpf.npoints
        @inbounds for i in get_idxs(ann[j])
            A[i, j] = 1
        end
    end
    # Construct graph
    g = gtype(A)
end

            
    
