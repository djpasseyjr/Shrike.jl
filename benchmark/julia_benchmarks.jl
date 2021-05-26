using Distances
using HDF5
using Shrike
using NearestNeighbors
using NPZ

#DATAFOLDER = "/pine/scr/d/j/djpassey/rpdata/"
#NAMES = ["fashionmnist", "mnist", "sift", "gist"]
#FNAMES = ["fashion-mnist-784-euclidean.hdf5", "mnist-784-euclidean.hdf5", "sift-128-euclidean.hdf5", "gist-960-euclidean.hdf5"]
DATAFOLDER = "/Users/djpassey/Downloads/"
NAMES = ["mist"]
FNAMES = ["mnist-784-euclidean.hdf5"]

println("Loading Datasets")
FILES = map(x -> string(DATAFOLDER, x), FNAMES)
SHRIKE_NTREES = [2]#, 5, 8, 10, 20, 100, 200]
SHRIKE_DEPTH = [4]#, 5, 7, 8, 9, 11, 12, 13]
SHRIKE_NVOTES = [1]#, 2, 3, 10, 20]
KDTREE_LEAFSIZE = [100]#, 300, 500, 1000, 2000, 5000, 10000]

struct NNData
    name::String
    fname::String
    nfeatures::Int
    npoints::Int
    ntest::Int
    k::Int
end

function NNData(name::String, fname::String)
    data = read(h5open(fname))
    nfeatures, npoints = size(data["train"])
    k = size(data["neighbors"], 1)
    ntest, _ = size(data["test"])
    return NNData(name, fname, nfeatures, npoints, ntest, k)
end

train(d::NNData) = read(h5open(d.fname))["train"]
test(d::NNData) = read(h5open(d.fname))["test"]
neighbors(d::NNData) = read(h5open(d.fname))["neighbors"]

brutenn(x::AbstractArray, X::AbstractArray, k::Int) =
    sortperm(map(i -> sqeuclidean(x, X[:,i]), 1:size(X,2)), alg=PartialQuickSort(k))[1:k]

allNNDatasets() = map(NNData, NAMES, FILES)

println("Datasets Loaded")
for nndata in allNNDatasets()
    println("Starting Dataset: $(nndata.name)")
    X = train(nndata)
    Xts = test(nndata)
    println("Data shape: $(size(X))")
    rpt_results::Array{Array{Float64, 1}, 1} = [[],[]]
    kdt_results::Array{Array{Float64, 1}, 1} = [[],[]]
    maxdepth = floor(log(2, nndata.npoints) - log(2, nndata.k))
    # Shrike Algorithm
    for ntrees in SHRIKE_NTREES
        for depth in SHRIKE_DEPTH[SHRIKE_DEPTH .< maxdepth]
            for nvotes in SHRIKE_NVOTES
                index = ShrikeIndex(X, nndata.k, depth=depth, ntrees=ntrees)
                tot_time = 0.0
                tot_recall = 0.0
                for i in 1:nndata.ntest
                    q = Xts[:, i]
                    tnn = brutenn(q, X, nndata.k)
                    td = @timed approx_nn = ann(index, q, nndata.k, vote_cutoff=nvotes)
                    tot_time += td.time
                    tot_recall += (length(intersect(td.value, tnn)) / nndata.k)
                end
                push!(rpt_results[1], nndata.ntest / tot_time)
                push!(rpt_results[2], tot_recall / nndata.ntest)
            end
        end
    end
    npzwrite("rpt-$(nndata.name).npy", hcat(rpt_results...))
    println("Shrike Complete")

    for leafsize in KDTREE_LEAFSIZE
        index = KDTree(X, leafsize=leafsize)
        tot_time = 0.0
        tot_recall = 0.0
        for i in 1:nndata.ntest
            q = Xts[:, i]
            tnn =  brutenn(q, X, nndata.k)
            td = @timed approxnn = knn(index, q, nndata.k)
            tot_time += td.time
            tot_recall += (length(intersect(td.value[1], tnn)) / nndata.k)
        end
        push!(kdt_results[1], nndata.ntest / tot_time)
        push!(kdt_results[2], tot_recall / nndata.ntest)
    end
    npzwrite("kdt-$(nndata.name).npy", hcat(kdt_results...))
    println("NearestNeighbors Complete")
end
