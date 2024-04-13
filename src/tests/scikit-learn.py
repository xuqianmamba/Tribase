from sklearn.neighbors import NearestNeighbors
import numpy as np
from timeit import default_timer as timer


def fvecs_read(filename, c_contiguous=True):
    fv = np.fromfile(filename, dtype=np.float32)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    fv = fv[:, 1:]
    if c_contiguous:
        fv = fv.copy()
    return fv


# ball-tree
base = fvecs_read("./benchmarks/msong/origin/msong_base.fvecs")
query = fvecs_read("./benchmarks/msong/origin/msong_query.fvecs")
nbrs = NearestNeighbors(n_neighbors=100, algorithm="ball_tree", n_jobs=-1).fit(base)

tic = timer()
distances, indices = nbrs.kneighbors(query)
toc = timer()
print("ball-tree: ", toc - tic)
