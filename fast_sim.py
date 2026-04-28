from tqdm.contrib.concurrent import process_map
from utils import sample_points_on_sphere
from octahedral import convex_hull_octahedron_test
import numpy as np
import pandas as pd
import time, datetime
import os

MAX_INT32 = int(2**32)
PATH = r"data"
TEMP_PATH = r"data\temp"


def _sim(seed, N):
    rng = np.random.default_rng(seed)
    results = 0

    for _ in range(N):
        xyz = sample_points_on_sphere(rng)
        regular = convex_hull_octahedron_test(xyz) # type: ignore
        results += regular

    return results

def par_sim(target=100_000):
    # Save seed for reproducibility
    seed = (time.time_ns() // 100) % MAX_INT32

    # Parallel Parameters:
    n_workers = os.cpu_count()
    batch_size = 10_000
    chunk_size = 1
    n_jobs = np.max([np.ceil(target / batch_size), 1]).astype(int)  # type: ignore

    # parameters for process_map
    rng = np.random.default_rng(seed)
    seed_list = rng.integers(1, 
                             MAX_INT32 - 1,
                             n_jobs)
    batch_sizes = [batch_size] * n_jobs

    # simulate at least target times
    r = process_map(_sim, seed_list, batch_sizes, max_workers=n_workers, chunksize=chunk_size)

    # save results
    data = np.array([r, seed_list, batch_sizes]).T
    results = pd.DataFrame(data, columns=["regular", "seed", "N"])

    print(f"n regular:     {results["regular"].sum()}\nn simulations: {results["N"].sum()}")
    p = results["regular"].sum() / results["N"].sum()
    print(f"{p=}")
    print(f"saving...")
    results.to_pickle(fr"{PATH}\results_{seed:010}.pkl")
    

if __name__ == "__main__":
    j = 0
    while True:
        m = 50
        for i in range(m):
            par_sim(1_000_000)
        
        j += 1
        f"generating data for {j} epochs."