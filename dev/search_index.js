var documenterSearchIndex = {"docs":
[{"location":"#RPTrees","page":"RPTrees","title":"RPTrees","text":"","category":"section"},{"location":"","page":"RPTrees","title":"RPTrees","text":"Create ensembles of random projection trees in Julia.","category":"page"},{"location":"#Functions","page":"RPTrees","title":"Functions","text":"","category":"section"},{"location":"","page":"RPTrees","title":"RPTrees","text":"RPForest(data::AbstractArray{T, 2}, maxdepth::Int, ntrees::Int) where T","category":"page"},{"location":"#RPTrees.RPForest-Union{Tuple{T}, Tuple{AbstractArray{T,2},Int64,Int64}} where T","page":"RPTrees","title":"RPTrees.RPForest","text":"RPForest(data::Array{T, 2}, maxdepth::Int, ntrees::Int) where T -> ensemble\n\nConstructor for ensemble of sparse random projection trees with voting. Follows the implementation outlined in:\n\nFast Nearest Neighbor Search through Sparse Random Projections and Voting. Ville Hyvönen, Teemu Pitkänen, Sotirios Tasoulis, Elias Jääsaari, Risto Tuomainen, Liang Wang, Jukka Ilmari Corander, Teemu Roos. Proceedings of the 2016 IEEE Conference on Big Data (2016)\n\n\n\n\n\n","category":"method"},{"location":"","page":"RPTrees","title":"RPTrees","text":"traverse_to_leaves(rpf::RPForest{T}, x::Array{T, 2}) where T","category":"page"},{"location":"#RPTrees.traverse_to_leaves-Union{Tuple{T}, Tuple{RPForest{T},Array{T,2}}} where T","page":"RPTrees","title":"RPTrees.traverse_to_leaves","text":"traverse_to_leaves(rpf::RPForest{T}, x::Array{T, 2}) where T -> leaf_idxs\n\nRoute data point x down to a leaf node each tree and return and array of indexes of the data stored in each corresponding leaf node\n\n\n\n\n\n","category":"method"}]
}
