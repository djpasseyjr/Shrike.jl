"""
    traverse_tree(shi::ShrikeIndex{T}, x::Array{T, 2}) where T -> leaf_idxs

Route data point `x` down to a leaf node each tree and return and array of
indexes of the data stored in each corresponding leaf node
"""
function traverse_tree(shi::ShrikeIndex{T}, x::AbstractArray{T, 2}) where T
    # Compute all needed projections
    proj::Array{T, 1} = reshape(transpose(x) * shi.random_vectors, :)
    # Initial node indexes (Root index for each tree)
    node_idxs = ones(Int, shi.ntrees)
    # Location of projection value for the current depth (Starts at
    # [1, shi.depth, 2*shi.depth, ... shi.depth*shi.ntrees].)
    projection_idxs::Array{Int, 1} = collect(1:shi.depth:shi.depth*shi.ntrees);
    # For each level of the trees
    @inbounds for d in 1:shi.depth
        # Determine if the projection is on the left or right of the split
        mask::Array{Bool, 1} = proj[projection_idxs] .>=  _get_splits(shi, node_idxs)
        # Move node indexes to the right child
        node_idxs .*= 2
        # Change to left child if the projection was less than the split value
        node_idxs[mask] .+= 1
        # Increase depth of projection locations by one
        projection_idxs .+= 1
    end
    # Subtract the number of non leaf nodes (Because we don't store indexes
    # at non leaf nodes)
    node_idxs .-= 2^shi.depth - 1
    leaf_idxs = map(i -> shi.indexes[node_idxs[i], i], 1:shi.ntrees)
    return leaf_idxs
end

traverse_tree(shi::ShrikeIndex{T}, x::AbstractArray{T, 1}) where T = traverse_tree(shi, reshape(x, :, 1))

"""
    _get_splits(shi::ShrikeIndex{T}, node_idx::Array{Int, 1}) where T -> splits

Given an array of node indexes of length `shi.ntrees` corresponding to the
current index in each tree, return the split values of each node.
"""
@inline _get_splits(shi::ShrikeIndex{T}, node_idx::Array{Int, 1}) where T = map(i -> shi.splits[i, node_idx[i]], 1:shi.ntrees)

"""
    leafsize(shi::ShrikeIndex) -> ls

Return the minimum number of points in the leaf nodes
"""
leafsize(shi::ShrikeIndex) = floor(Int, shi.npoints / 2^shi.depth)

"""
    getdepth(leafsize::Int, npoints::Int) -> d

Returns the depth needed for the tree to have approximately `leafsize` points in the
leaf nodes.
"""
getdepth(leafsize::Int, npoints::Int) = floor(Int, log(2, npoints) - log(2, leafsize))

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

@inline function Base.push!(ne::NeighborExplorer{T}, idx::Int, dist::T) where T
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
@inline function Base.pop!(ne::NeighborExplorer{T}) where T
    # Keep seen indexes in the index set so we don't consider them again
    maxdist = pop!(ne.heap)
    ne.npoints -= 1
    return maxdist
end

"""
    get_dists(ne::NeighborExplorer) -> neighbor_idxs

Return the current distances of `k` nearest neighbors.
"""
@inline get_dists(ne::NeighborExplorer) = [dist for (dist, idx) in ne.heap.valtree]

"""
    get_idxs(ne::NeighborExplorer) -> neighbor_idxs

Return the current indicies to `k` nearest neighbors.
"""
@inline get_idxs(ne::NeighborExplorer) = [idx for (dist, idx) in ne.heap.valtree]


function Base.show(io::IO, ne::NeighborExplorer)
    descr = "NeighborExplorer: \n    Currently holds $(ne.npoints)/$(ne.k) points \n    $(length(ne.idx_set)) points already explored "
    print(io, descr)
end

""" Helper function for RP-Tree voting """
@inline vote!(v::Array{UInt16, 1}, idx::Array{Int, 1}) = v[idx] .+= 0x0001

"""
    candidate_idxs(shi::ShrikeIndex{T}, q::Array{T, N}, k::Int; vote_cutoff=1) where {T, N} -> cand_idx

Each RP-Tree in the `ShrikeIndex` "votes" for nearest neighbor candidates. All points that receive
more than `vote_cutoff` votes are returned. By default `vote_cutoff=1` and this returns
the set union of all RP-Tree leaf nodes in the ensemble. This is the standard RP-Tree algorithm.
"""
@inline function candidate_idxs(shi::ShrikeIndex{T}, q::AbstractArray{T, N}, k::Int; vote_cutoff=1) where {T, N}
    leaf_idxs = traverse_tree(shi, q)
    # Empty array of votes
    votes = zeros(UInt16, shi.npoints)
    # Each tree votes on candiate nearest neighbors
    ThreadsX.map(idxs -> vote!(votes, idxs), leaf_idxs)
    # Find candiate points with enough votes
    vote_mask = votes .>= UInt16(vote_cutoff)
	cand_idx::Array{Int, 1} = findall(vote_mask)
    return cand_idx
end

"""
    approx_knn(shi::ShrikeIndex{T}, q::Array{T, 2}, k::Int; vote_cutoff=1) where T -> knn_idx

For a query point `q`, find the approximate `k` nearest neighbors from the data stored in the the
ShrikeIndex. The `vote_cutoff` parameter signifies how many "votes" a point needs in order to be included
in a linear search. Increasing `vote_cutoff` speeds up the algorithm but may reduce accuracy.

"""
function ann(shi::ShrikeIndex{T}, q::AbstractArray{T, N}, k::Int; vote_cutoff::Int=1) where {T, N}
	safe_vote_cutoff = min(shi.ntrees, vote_cutoff)
    cand_idx = candidate_idxs(shi, q, k, vote_cutoff=safe_vote_cutoff)
    ncand::Int = length(cand_idx)
    cand_dist = ThreadsX.map(i -> disttopoint(i, q, shi.data), cand_idx)
    sp = sortperm(cand_dist, alg=PartialQuickSort(k))
	@inbounds knn_idx = [cand_idx[sp[i]] for i=1:min(ncand,k)]
    return knn_idx
end

""" Distance between points in a set (Helper function for threading)
"""
@inline function disttopoint(i::Int, q::AbstractArray{T, N}, X::AbstractArray{T, 2}) where {T, N}
    @inbounds x = @view X[:, i]
    return sqeuclidean(x, q)
end

"""
    vote!(vs::Array{Dict{Int,Int}, 1}, idxs::Array{Int, 1})

Helper function for creating a knn-graph from a ShrikeIndex
"""
@inline function vote!(vs::Array{Dict{Int,UInt16}, 1}, idxs::Array{Int, 1})
    @inbounds for i in idxs
         for j in idxs
            v = get(vs[i], j, 0x123)
            vs[i][j] = v + 0x0001
            u = get(vs[j], i, 0x123)
            vs[j][i] = u + 0x0001
        end
    end
end

"""
    collect_votes(shi::ShrikeIndex{T}, k::Int) -> votes

Iterate through each leaf and collect predicted neighbors. Count how often
two nodes appear together in leaf nodes. (These are the votes.) We can
filter candidate neighbors by number of votes later.
"""
function collect_votes(shi::ShrikeIndex{T}, k::Int) where T
	votes = [Dict{Int,UInt16}() for i in 1:shi.npoints]
    nleafs = 2^shi.depth
    for i in 1:shi.ntrees
        for j in 1:nleafs
            vote!(votes, shi.indexes[j, i])
        end
    end
	return votes
end

"""
    build_neighbor_explorer(i::Int, votes::Dict{Int, UInt16}, shi::ShrikeIndex{T}, k::Int, vote_cutoff::Int) -> approx_nn

Linear search through candidate nearest neighbors. Store in a heap.
"""
function build_neighbor_explorer(
	i::Int,
	votes::Dict{Int, UInt16},
	shi::ShrikeIndex{T},
	k::Int,
	vote_cutoff::Int
) where T
	approx_nn = NeighborExplorer{T}(i, k)
	nodes = collect(keys(votes))
	@inbounds x = @view shi.data[:, i]
	@inbounds for node in nodes
		if votes[node] >= vote_cutoff
			y = @view shi.data[:, node]
			push!(approx_nn, node, sqeuclidean(x,y))
		end
	end
	return approx_nn
end

"""
    _allknn(shi::ShrikeIndex{T}, k::Int; vote_cutoff=1) where T -> knn

Returns a neighbor explorer for each point to allow for neighbors of neighbors exploration
and the creation of a knn-graph.

The `vote_cutoff` parameter signifies how many "votes" a point needs in order to be
included in a linear search. Increasing `vote_cutoff` speeds up the algorithm but may reduce accuracy.
"""
function _allknn(shi::ShrikeIndex{T}, k::Int, vote_cutoff::Int) where T
	votes = collect_votes(shi, k)
    knn = ThreadsX.map(
		(i,v) -> build_neighbor_explorer(i, v, shi, k, vote_cutoff),
		1:shi.npoints,
		votes
	)
    return knn
end

"""
    _allcandidates(shi::ShrikeIndex{T}, k::Int) -> allneighbors

Optimizations for the case when `vote_cutoff=1`. Use a set union of the leaf
indexes instead of a dictionary of votes.
"""
function _allcandidates(shi::ShrikeIndex{T}, k::Int) where T
	nleafs = 2^shi.depth
    allneighbors = [Set{Int64}() for i=1:shi.npoints]
	for i in 1:shi.ntrees
		for j in 1:nleafs
			ind = shi.indexes[j, i]
			for k in ind
				indset = allneighbors[k]
				for l in ind
					push!(indset, l)
				end
			end
		end
	end
	return allneighbors
end

"""
    build_neighbor_explorer(i::Int, candidates::Set{Int}, shi::ShrikeIndex{T}, k::Int) -> approx_nn

Optimizations for the case when `vote_cutoff=1`. Don't check number of votes.
Iterate though set of candidates.
"""
function build_neighbor_explorer(
	i::Int,
	candidates::Set{Int},
	shi::ShrikeIndex{T},
	k::Int
) where T
	approx_nn = NeighborExplorer{T}(i, k)
	@inbounds x = @view shi.data[:, i]
	 for node in candidates
		@inbounds y = @view shi.data[:, node]
		push!(approx_nn, node, sqeuclidean(x,y))
	end
	return approx_nn
end

"""
	_allknn(shi::ShrikeIndex{T}, k::Int) -> approx_nns

Optimizations for finding the nearest neighbors of every point in
the case `vote_cutoff=1`
"""
function _allknn(shi::ShrikeIndex{T}, k::Int) where T
	allneighbors = _allcandidates(shi, k)
	approx_nns = ThreadsX.map(
		(i,v) -> build_neighbor_explorer(i, v, shi, k),
		1:shi.npoints,
		allneighbors
	)
	return approx_nns
end

"""
    explore!(data::Array{T, 2}, approx_nn::Array{NeighborExplorer{T}, 1}) where T

Explore neighbors of neighbors to improve accuracy of the approximate
k-nearest-neigbors stored in `knn`.

**Parameters**
1. `i` is the current datapoint
2. `data` is the array of data. Each column is a data point
3. `approx_nn` is an array of `NeighborExplorer`s where `approx_nn[i]` contains
the current best approximation to the k-nearest-neigbors of point `i`.

"""
function explore(i::Int, data::AbstractArray{T}, approx_nn::Array{NeighborExplorer{T}, 1}) where T
    x = @view data[:, i]
    new_approx_nn = deepcopy(approx_nn[i])
    for j in get_idxs(approx_nn[i])
            for l in get_idxs(approx_nn[j])
                y = @view data[:, l:l]
                dist = sqeuclidean(x, y)
                push!(new_approx_nn, l, dist)
        end
    end
    return new_approx_nn
end

"""
    allknn(shi::ShrikeIndex{T}, k::Int; vote_cutoff::Int=1, ne_iters::Int=0) where T -> approxnn_array

Returns a `shi.npoints` by `k` array of approximate nearest neighbor indexes.
That is, `approxnn_array[i,:]` contains the indexes of the k nearest neighbors of
`shi.data[:, i]`.

**Parameters**

1. The `ne_iters` assigns the number of iterations of neighbor exploration to use.
Neighbor exploration is an inexpensive way to increase accuracy.

2, The `vote_cutoff` parameter signifies how many "votes" a point needs in order to be included
in a linear search. Increasing `vote_cutoff` speeds up the algorithm but may reduce accuracy.
Passing too large of a `vote_cutoff` results in the algorithm resetting `vote_cutoff` to equal the number of trees.
"""
function allknn(shi::ShrikeIndex{T}, k::Int; vote_cutoff::Int=1, ne_iters::Int=0) where T
	safe_vote_cutoff = min(shi.ntrees, vote_cutoff)
    # Search tree via neighbor explorers
	if safe_vote_cutoff == 1
		approx_nns = _allknn(shi, k)
	else
    	approx_nns = _allknn(shi, k, safe_vote_cutoff)
	end
    # Neighbor exploration
    for i in 1:ne_iters
        approx_nns = ThreadsX.map(i -> explore(i, shi.data, approx_nns), 1:shi.npoints)
    end
    # Load neighbor explorers into a matrix
    approxnn_array = zeros(Int, shi.npoints, k)
    for (i, ne) in enumerate(approx_nns)
        @inbounds approxnn_array[i, :] = get_idxs(ne)
    end
    return approxnn_array
end

"""
    add_edges!(j::Int, approx_nnj::NeighborExplorer{T}, A::AbstractArray{U, 2}) where {T, U}

Helper function for multithreading knn-graph adjacency matrix creation
"""
function add_edges!(j::Int, approx_nnj::NeighborExplorer{T}, A::AbstractArray{U, 2}) where {T, U}
     for i in get_idxs(approx_nnj)
        @inbounds A[i, j] = 1
    end
end

"""
    knngraph(shi::ShrikeIndex{T}, k::Int, vote_cutoff; vote_cutoff::Int=1, ne_iters::Int=0, gtype::G) where {T, G} -> g

Returns a graph with `shi.npoints` node and `k * shi.npoints` edges datapoints conneceted to nearest neighbors

**Parameters**

1. `shi`: random forest of the desired data
2. `k`: the desired number of nearest neighbors
3. `vote_cutoff`: signifies how many "votes" a point needs in order to be included
in a linear search through leaf nodes. Increasing `vote_cutoff` speeds up the algorithm but may reduce accuracy.
Passing too large of a `vote_cutoff` results in the algorithm resetting `vote_cutoff` to equal the number of trees.
4. `ne_iters`: assigns the number of iterations of neighbor exploration to use. Defaults to zero.
Neighbor exploration is a way to increse knn-graph accuracy.
5. `gtype` is the type of graph to construct. Defaults to `SimpleDiGraph`. `gtype=identity` returns a sparse adjacency matrix.

"""
function knngraph(
	shi::ShrikeIndex{T},
	k::Int;
	vote_cutoff::Int=1,
	ne_iters::Int=0,
	gtype::G=SimpleDiGraph
) where {T, G}
    # Search tree via neighbor explorers
	if vote_cutoff == 1
		approx_nns = _allknn(shi, k)
	else
    	approx_nns = _allknn(shi, k, vote_cutoff)
	end
    # Neighbor exploration (In parallel)
    for i in 1:ne_iters
        approx_nns = ThreadsX.map(i -> explore(i, shi.data, approx_nns), 1:shi.npoints)
    end
    # Load neighbor explorers into a sparse adj matrix (TODO: In parallel)
    A = spzeros(Int, shi.npoints, shi.npoints)
    for j in 1:shi.npoints
        add_edges!(j, approx_nns[j], A)
    end
    # Construct graph
    g = gtype(A)
end
