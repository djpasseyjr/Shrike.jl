var documenterSearchIndex = {"docs":
[{"location":"#RPTrees","page":"RPTrees","title":"RPTrees","text":"","category":"section"},{"location":"","page":"RPTrees","title":"RPTrees","text":"Create ensembles of random projection trees in Julia.","category":"page"},{"location":"#Functions","page":"RPTrees","title":"Functions","text":"","category":"section"},{"location":"","page":"RPTrees","title":"RPTrees","text":"RPForest(data::AbstractArray{T, 2}, maxdepth::Int, ntrees::Int) where T","category":"page"},{"location":"#RPTrees.RPForest-Union{Tuple{T}, Tuple{AbstractMatrix{T}, Int64, Int64}} where T","page":"RPTrees","title":"RPTrees.RPForest","text":"RPForest(data::Array{T, 2}, maxdepth::Int, ntrees::Int) where T -> ensemble\n\nConstructor for ensemble of sparse random projection trees with voting. Follows the implementation outlined in:\n\nFast Nearest Neighbor Search through Sparse Random Projections and Voting. Ville Hyvönen, Teemu Pitkänen, Sotirios Tasoulis, Elias Jääsaari, Risto Tuomainen, Liang Wang, Jukka Ilmari Corander, Teemu Roos. Proceedings of the 2016 IEEE Conference on Big Data (2016)\n\n\n\n\n\n","category":"method"},{"location":"","page":"RPTrees","title":"RPTrees","text":"knngraph(rpf::RPForest{T}, k::Int; vote_cutoff::Int=1, ne_iters::Int=0, gtype::G=SimpleDiGraph) where {T, G}","category":"page"},{"location":"#RPTrees.knngraph-Union{Tuple{G}, Tuple{T}, Tuple{RPForest{T}, Int64}} where {T, G}","page":"RPTrees","title":"RPTrees.knngraph","text":"knngraph(rpf::RPForest{T}, k::Int, vote_cutoff; vote_cutoff::Int=1, ne_iters::Int=0, gtype::G) where {T, G} -> g\n\nReturns a graph with rpf.npoints node and k * rpf.npoints edges datapoints conneceted to nearest neighbors\n\nParameters\n\nrpf: random forest of the desired data\nk: the desired number of nearest neighbors\nvote_cutoff: signifies how many \"votes\" a point needs in order to be included \n\nin a linear search through leaf nodes. Increasing vote_cutoff speeds up the algorithm but may reduce accuracy. Defaults to 1\n\nne_iters: assigns the number of iterations of neighbor exploration to use. Defaults to zero.\n\nNeighbor exploration is a way to increse knn-graph accuracy.\n\ngtype is the type of graph to construct. Defaults to SimpleDiGraph\n\n\n\n\n\n","category":"method"},{"location":"","page":"RPTrees","title":"RPTrees","text":"approx_knn(rpf::RPForest{T}, q::AbstractArray{T, 2}, k::Int; vote_cutoff=1) where T","category":"page"},{"location":"#RPTrees.approx_knn-Union{Tuple{T}, Tuple{RPForest{T}, AbstractMatrix{T}, Int64}} where T","page":"RPTrees","title":"RPTrees.approx_knn","text":"approx_knn(rpf::RPForest{T}, q::Array{T, 2}, k::Int; vote_cutoff=1) where T -> knn_idx\n\nFor a query point q, find the approximate k nearest neighbors from the data stored in the the RPForest. The vote_cutoff parameter signifies how many \"votes\" a point needs in order to be included in a linear search. Increasing vote_cutoff speeds up the algorithm but may reduce accuracy.\n\n\n\n\n\n","category":"method"},{"location":"","page":"RPTrees","title":"RPTrees","text":"allknn(rpf::RPForest{T}, k::Int; vote_cutoff::Int=1, ne_iters::Int=0) where T","category":"page"},{"location":"#RPTrees.allknn-Union{Tuple{T}, Tuple{RPForest{T}, Int64}} where T","page":"RPTrees","title":"RPTrees.allknn","text":"allknn(rpf::RPForest{T}, k::Int; vote_cutoff::Int=1, ne_iters::Int=0) where T -> approxnn\n\nReturns a rpf.npoints by k array of approximate nearest neighbor indexes. That is, approxnn[i,:] contains the indexes of the k nearest neighbors of rpf.data[:, i].\n\nThe ne_iters assigns the number of iterations of neighbor exploration to use.  Neighbor exploration is an inexpensive way to increase accuracy.\n\nThe vote_cutoff parameter signifies how many \"votes\" a point needs in order to be included  in a linear search. Increasing vote_cutoff speeds up the algorithm but may reduce accuracy.\n\n\n\n\n\n","category":"method"},{"location":"","page":"RPTrees","title":"RPTrees","text":"traverse_to_leaves(rpf::RPForest{T}, x::Array{T, 2}) where T","category":"page"},{"location":"#RPTrees.traverse_to_leaves-Union{Tuple{T}, Tuple{RPForest{T}, Matrix{T}}} where T","page":"RPTrees","title":"RPTrees.traverse_to_leaves","text":"traverse_to_leaves(rpf::RPForest{T}, x::Array{T, 2}) where T -> leaf_idxs\n\nRoute data point x down to a leaf node each tree and return and array of indexes of the data stored in each corresponding leaf node\n\n\n\n\n\n","category":"method"}]
}
