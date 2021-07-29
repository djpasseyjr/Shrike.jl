using Shrike
using Shrike: traverse_tree, getdepth, leafsize, _allknn, get_dists
using Test

@testset "Constructor" begin
    X = rand(250, 16)
    shi = ShrikeIndex(X, 3, 3)
    # Check for self consistency
    @test all((size(shi.splits) .== (shi.ntrees, 2^shi.depth - 1)))
    @test all(size(shi.indexes) .== (2^shi.depth, shi.ntrees))
    @test all(size(shi.random_vectors) .== (shi.ndims, shi.ntrees * shi.depth))

    shi = ShrikeIndex(X, depth=5, ntrees=3)
    # Check for self consistency
    @test all((size(shi.splits) .== (shi.ntrees, 2^shi.depth - 1)))
    @test all(size(shi.indexes) .== (2^shi.depth, shi.ntrees))
    @test all(size(shi.random_vectors) .== (shi.ndims, shi.ntrees * shi.depth))

    shi = ShrikeIndex(X, 3)
    # Check for self consistency
    @test all((size(shi.splits) .== (shi.ntrees, 2^shi.depth - 1)))
    @test all(size(shi.indexes) .== (2^shi.depth, shi.ntrees))
    @test all(size(shi.random_vectors) .== (shi.ndims, shi.ntrees * shi.depth))

    shi = ShrikeIndex(X, 3, depth=3, ntrees=10)
    # Check for self consistency
    @test all((size(shi.splits) .== (shi.ntrees, 2^shi.depth - 1)))
    @test all(size(shi.indexes) .== (2^shi.depth, shi.ntrees))
    @test all(size(shi.random_vectors) .== (shi.ndims, shi.ntrees * shi.depth))

end


# Check that for each data point used to build the tree
# traverse_tree returns leaf nodes containing the given datapoint
@testset "Traverse" begin
    X = rand(250, 16)
    shi = ShrikeIndex(X, 3, 3)
    for j =1:size(X)[2]
        @test all(map(idxs -> j in idxs, traverse_tree(shi, reshape(X[:, j], :, 1))))
    end
end

@testset "Nearest Neighbors" begin
    k = 10
    npoints = 1000
    X = randn(300, npoints)
    shi = ShrikeIndex(X)
    @test length(ann(shi, X[:, 1:1], k; vote_cutoff=1)) == k
    @test all(size(allknn(shi, k, ne_iters=1)) .== (npoints, k))
    @test (knngraph(shi, k, ne_iters=1)).ne == k * npoints
    shi100 = ShrikeIndex(X, ntrees=100)
    @test length(ann(shi, X[:, 1:1], k; vote_cutoff=2)) == k
    @test all(size(allknn(shi, k, ne_iters=2)) .== (npoints, k))
    @test (knngraph(shi, k, ne_iters=2)).ne == k * npoints
    @test (knngraph(shi, k, ne_iters=2, vote_cutoff=2)).ne == k * npoints        
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
