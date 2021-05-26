using Shrike
using Shrike: traverse_tree, getdepth, leafsize, _allknn, get_dists
using Test

@testset "Constructor" begin
    X = rand(250, 16)
    rpf = ShrikeIndex(X, 3, 3)
    # Check for self consistency
    @test all((size(rpf.splits) .== (rpf.ntrees, 2^rpf.depth - 1)))
    @test all(size(rpf.indexes) .== (2^rpf.depth, rpf.ntrees))
    @test all(size(rpf.random_vectors) .== (rpf.ndims, rpf.ntrees * rpf.depth))

    rpf = ShrikeIndex(X, depth=5, ntrees=3)
    # Check for self consistency
    @test all((size(rpf.splits) .== (rpf.ntrees, 2^rpf.depth - 1)))
    @test all(size(rpf.indexes) .== (2^rpf.depth, rpf.ntrees))
    @test all(size(rpf.random_vectors) .== (rpf.ndims, rpf.ntrees * rpf.depth))

    rpf = ShrikeIndex(X, 3)
    # Check for self consistency
    @test all((size(rpf.splits) .== (rpf.ntrees, 2^rpf.depth - 1)))
    @test all(size(rpf.indexes) .== (2^rpf.depth, rpf.ntrees))
    @test all(size(rpf.random_vectors) .== (rpf.ndims, rpf.ntrees * rpf.depth))

end


# Check that for each data point used to build the tree
# traverse_tree returns leaf nodes containing the given datapoint
@testset "Traverse" begin
    X = rand(250, 16)
    rpf = ShrikeIndex(X, 3, 3)
    for j =1:size(X)[2]
        @test all(map(idxs -> j in idxs, traverse_tree(rpf, reshape(X[:, j], :, 1))))
    end
end

@testset "Nearest Neighbors" begin
    k = 10
    npoints = 1000
    X = randn(300, npoints)
    rpf = ShrikeIndex(X)
    @test length(ann(rpf, X[:, 1:1], k; vote_cutoff=1)) == k
    @test all(size(allknn(rpf, k, ne_iters=1)) .== (npoints, k))
    @test (knngraph(rpf, k, ne_iters=1)).ne == k * npoints
    rpf100 = ShrikeIndex(X, ntrees=100)
    @test length(ann(rpf, X[:, 1:1], k; vote_cutoff=2)) == k
    @test all(size(allknn(rpf, k, ne_iters=2)) .== (npoints, k))
    @test (knngraph(rpf, k, ne_iters=2)).ne == k * npoints
end



@testset "Utils" begin
    l = 10
    k = 5
    npoints = 1010
    X = randn(10, npoints)
    @test leafsize(ShrikeIndex(X, depth=getdepth(l, 100))) >= l
    ne = _allknn(ShrikeIndex(X, ntrees=2), k)
    @test length(get_dists(ne[1])) == k
end
