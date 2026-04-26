import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from utils import sample_points_on_sphere
from octahedral import is_octahedral
from multiprocessing import Pool
import time, datetime


edge_dtype = np.dtype([('u', np.uint8), ('v', np.uint8)])
PATH = r"data"
TEMP_PATH = r"data\temp"


def sim(seed, N, save_edges=False, save_points=False):
    rng = np.random.default_rng(seed)
    edge_names = [f"e{i}" for i in range(1, 12+1)]
    
    XYZ = np.empty((N, 6 * 3), dtype=np.float64) if save_points else None
    results = np.zeros((N, 1), dtype=np.byte)
    edge_list = np.empty((N, len(edge_names)), dtype=object) if save_edges else None

    for i in range(N):
        xyz = sample_points_on_sphere()

        # TODO: change type ignore -> more pythonic
        results[i], edges = is_octahedral(xyz) # type: ignore
        
        if save_edges: edge_list[i, :] = np.array([*edges], dtype=edge_dtype) # type: ignore
        if save_points: XYZ[i, :] = xyz.reshape(1, -1) # type: ignore

    results = pd.DataFrame(data=np.hstack([results, edge_list]) if save_edges else results, columns=["regular", *(edge_names if save_edges else [])])

    return results, XYZ


def batched_sim():
    seed = (time.time_ns() // 1000) % 10_000
    n_workers = 24
    batch_size = 1_000
    n_jobs = n_workers * 50
    rng = np.random.default_rng(seed)
    seed_list = rng.integers(100, 
                             10_000, # arbitrary
                             n_jobs)
    batch_sizes = [batch_size] * n_jobs

    with Pool(n_workers) as p:
        batches = p.starmap(sim, zip(seed_list, batch_sizes), chunksize=8)

    _results, _xyz = list(zip(*batches))

    results = pd.concat(_results, axis=0)
    xyz     = np.vstack(_xyz)

    results.to_pickle(fr"{PATH}\results_{seed:04}_{time.time_ns()}.pkl")
    np.save(fr"{PATH}\xyz", xyz, allow_pickle=True)


if __name__ == "__main__":
    start = time.time()

    batched_sim()
    
    print(time.time() - start)