import h5py
#import mrpt
import annoy
import pickle as pkl
import numpy as np
import time
import threading
print(f"Active threads: {threading.active_count()}")

# DATAFOLDER = "/pine/scr/d/j/djpassey/rpdata/"
# NAMES = ["fashionmnist", "sift", "gist"]
# FNAMES = ["fashion-mnist-784-euclidean.hdf5", "mnist-784-euclidean.hdf5", "sift-128-euclidean.hdf5", "gist-960-euclidean.hdf5"]

DATAFOLDER = "/Users/djpassey/Downloads/"
NAMES = ["mist"]
FNAMES = ["mnist-784-euclidean.hdf5"]
FILES = [DATAFOLDER + f for f in FNAMES]


class NNData:
    def __init__(self, name, fname):
        data = h5py.File(fname, 'r')
        m, n = data["train"].shape
        mts, nts = data["test"].shape
        k = data["neighbors"].shape[0]
        self.name = name
        self.fname = fname
        self.nfeatures = n
        self.npoints = m
        self.ntest = nts
        self.k = k

    def train(self):
        return h5py.File(self.fname, 'r')['train'][:, :]
    def test(self):
        return h5py.File(self.fname, 'r')['test'][:, :]
    def neighbors(self):
        return h5py.File(self.fname, 'r')['neighbors'][:, :]

NNDATASETS = [NNData(n, f) for n,f in zip(NAMES, FILES)]
MRPT_TARGET_RECALLS = [0.5]#, 0.6, 0.7 0.8 0.9 0.95, 0.97 0.99]
ANNOY_NTREES = [1]#, 2, 5, 10, 20, 50, 100, 200]
ANNOY_SEARCHK = [-1]#, 100, 300, 500, 1000, 2000, 5000, 10000]

def recall(ann, true_nn):
    return len(set(ann).intersection(set(true_nn))) / len(true_nn)

def dist(x,y):
    return np.sum((x - y)**2)

def brutenn(x, X, k):
    n = X.shape[0]
    distances = [dist(x, X[i, :]) for i in range(n)]
    ind = np.argsort(distances)[1:k]
    return list(ind)

print("Datasets Loaded")
for nndata in NNDATASETS:
    print(f"Starting Dataset: {nndata.name}")
    data = nndata.train()
    test_data = nndata.test()
    mrpt_results = [[],[]]
    annoy_results = [[],[]]

    for nt in ANNOY_NTREES:
        for sk in ANNOY_SEARCHK:
            index = annoy.AnnoyIndex(nndata.nfeatures, metric='euclidean')
            for i in range(nndata.npoints):
                v = data[i, :]
                index.add_item(i, v)
            index.build(nt)
            tot_time = 0.0
            tot_recall = 0.0
            for i in range(nndata.ntest):
                q = test_data[i, :]
                tnn = brutenn(q, data, nndata.k)
                t0 = time.time()
                ann = index.get_nns_by_vector(q, nndata.k, search_k=sk)
                tot_time += time.time() - t0
                tot_recall += recall(ann, tnn)
            qps = nndata.ntest / tot_time
            mean_recall = tot_recall / nndata.ntest
            annoy_results[0].append(qps)
            annoy_results[1].append(mean_recall)
    np.save(f"annoy-{nndata.name}.npy", np.array(annoy_results))
    print("Annoy Complete")

    # MRPT Algorithm
    for tr in MRPT_TARGET_RECALLS:
        index = mrpt.MRPTIndex(data)
        index.build_autotune_sample(tr, nndata.k)
        tot_time = 0.0
        tot_recall = 0.0
        for i in range(nndata.ntest):
            q = test_data[i, :]
            tnn = brutenn(q, data, nndata.k)
            t0 = time.time()
            ann = index.ann(q, return_distances=False)
            tot_time += time.time() - t0
            tot_recall += recall(ann, tnn)
        qps = nndata.ntest / tot_time
        mean_recall = tot_recall / nndata.ntest
        mrpt_results[0].append(qps)
        mrpt_results[1].append(mean_recall)
    np.save(f"mrpt-{nndata.name}.npy", np.array(mrpt_results))
    print("MRPT Complete")
