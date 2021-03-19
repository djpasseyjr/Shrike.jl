using RPTrees
using RPTrees: traverse_to_leaves

X = rand(250, 16)
rpf = RPForest(X, 3, 3)
# Check for self consistency
@assert all((size(rpf.splits) .== (rpf.ntrees, 2^rpf.depth - 1)))
@assert all(size(rpf.indexes) .== (2^rpf.depth, rpf.ntrees))
@assert all(size(rpf.random_vectors) .== (rpf.ndims, rpf.ntrees * rpf.depth))


# Check that for each data point used to build the tree
# traverse_to_leaves returns leaf nodes containing the given datapoint

for j =1:size(X)[2]
    @assert all(map(idxs -> j in idxs, traverse_to_leaves(rpf, reshape(X[:, j], :, 1))))
end
