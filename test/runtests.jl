using RPTrees
using RPTrees: traverse_to_leaves, getdepth, leafsize, _allknn, get_dists
using Test

@testset "Constructor" begin
    X = rand(250, 16)
    rpf = RPForest(X, 3, 3)
    # Check for self consistency
    @test all((size(rpf.splits) .== (rpf.ntrees, 2^rpf.depth - 1)))
    @test all(size(rpf.indexes) .== (2^rpf.depth, rpf.ntrees))
    @test all(size(rpf.random_vectors) .== (rpf.ndims, rpf.ntrees * rpf.depth))
end


# Check that for each data point used to build the tree
# traverse_to_leaves returns leaf nodes containing the given datapoint
@testset "Traverse" begin
    X = rand(250, 16)
    rpf = RPForest(X, 3, 3)
    for j =1:size(X)[2]
        @test all(map(idxs -> j in idxs, traverse_to_leaves(rpf, reshape(X[:, j], :, 1))))
    end
end

@testset "Nearest Neighbors" begin
    k = 10
    npoints = 1000
    X = randn(300, npoints)
    rpf = RPForest(X)
    @test length(approx_knn(rpf, X[:, 1:1], k; vote_cutoff=1)) == k
    @test all(size(allknn(rpf, k, ne_iters=1)) .== (npoints, k))
    @test (knngraph(rpf, k, ne_iters=1)).ne == k * npoints
end

@testset "Utils" begin
    l = 10
    k = 5
    npoints = 1010
    X = randn(10, npoints)
    @test leafsize(RPForest(X, depth=getdepth(l, 100))) >= l
    ne = _allknn(RPForest(X, ntrees=2), k)
    @test length(get_dists(ne[1])) == k
end