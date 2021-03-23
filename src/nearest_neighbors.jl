"""
    traverse_to_leaves(rpf::RPForest{T}, x::Array{T, 2}) where T -> leaf_idxs

Route data point `x` down to a leaf node each tree and return and array of
indexes of the data stored in each corresponding leaf node
"""
function traverse_to_leaves(rpf::RPForest{T}, x::Array{T, 2}) where T
    #TODO Dispatch on shape of x?
    # Compute all needed projections
    proj = reshape(transpose(x) * rpf.random_vectors, :)
    # Initial node indexes (Root index for each tree)
    node_idxs = ones(Int, rpf.ntrees)
    # Location of projection value for the current depth (Starts at
    # [1, rpf.depth, 2*rpf.depth, ... rpf.depth*rpf.ntrees].)
    projection_idxs = collect(1:rpf.depth:rpf.depth*rpf.ntrees);
    # For each level of the trees
    for d in 1:rpf.depth
        # Determine if the projection is on the left or right of the split
        mask = proj[projection_idxs] .>=  _get_splits(rpf, node_idxs)
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
