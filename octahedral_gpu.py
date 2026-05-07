import torch
import utils
import time
import pandas as pd

def make_hull_constants(device):
    # All C(6, 3) candidate triangular faces
    tri = torch.tensor([
        [0, 1, 2], [0, 1, 3], [0, 1, 4], [0, 1, 5],
        [0, 2, 3], [0, 2, 4], [0, 2, 5],
        [0, 3, 4], [0, 3, 5],
        [0, 4, 5],
        [1, 2, 3], [1, 2, 4], [1, 2, 5],
        [1, 3, 4], [1, 3, 5],
        [1, 4, 5],
        [2, 3, 4], [2, 3, 5],
        [2, 4, 5],
        [3, 4, 5],
    ], device=device, dtype=torch.long)

    # All C(6, 2) possible edges
    edge_pairs = torch.tensor([
        [0, 1], [0, 2], [0, 3], [0, 4], [0, 5],
        [1, 2], [1, 3], [1, 4], [1, 5],
        [2, 3], [2, 4], [2, 5],
        [3, 4], [3, 5],
        [4, 5],
    ], device=device, dtype=torch.long)

    # Mask: candidate face contains vertex m?
    in_tri = torch.zeros((20, 6), device=device, dtype=torch.bool)
    in_tri.scatter_(1, tri, True)

    # Mask: candidate triangular face contains candidate edge?
    tri_has_edge = (
        (tri[:, None, :] == edge_pairs[None, :, 0, None]).any(dim=-1)
        &
        (tri[:, None, :] == edge_pairs[None, :, 1, None]).any(dim=-1)
    )

    # Incidence matrix: edge -> vertices
    incidence = torch.zeros((15, 6), device=device, dtype=torch.int64)
    incidence.scatter_(1, edge_pairs[:, 0:1], 1)
    incidence.scatter_(1, edge_pairs[:, 1:2], 1)

    return tri, edge_pairs, in_tri, tri_has_edge, incidence


def convex_hull_edge_mask_gpu(points, eps=1e-9, constants=None):
    """
    GPU tensor version.

    points:
        shape (6, 3) or (B, 6, 3), on CUDA

    Returns:
        edge_pairs: tensor of shape (15, 2)
        edge_present: tensor of shape (15,) or (B, 15)
    """
    squeeze = points.ndim == 2
    if squeeze:
        points = points[None, :, :]

    device = points.device
    if constants is None:
        constants = make_hull_constants(device)

    tri, edge_pairs, in_tri, tri_has_edge, _ = constants

    p0 = points[:, tri[:, 0], :]
    p1 = points[:, tri[:, 1], :]
    p2 = points[:, tri[:, 2], :]

    normals = torch.cross(p1 - p0, p2 - p0, dim=-1)

    nondegenerate = torch.linalg.vector_norm(normals, dim=-1) >= eps

    # Signed distances of all 6 points from each candidate face plane
    d = torch.sum(
        normals[:, :, None, :] * (points[:, None, :, :] - p0[:, :, None, :]),
        dim=-1,
    )

    other = ~in_tri

    has_pos = ((d > eps) & other[None, :, :]).any(dim=-1)
    has_neg = ((d < -eps) & other[None, :, :]).any(dim=-1)

    is_face = nondegenerate & ~(has_pos & has_neg)

    edge_present = (is_face[:, :, None] & tri_has_edge[None, :, :]).any(dim=1)

    if squeeze:
        edge_present = edge_present[0]

    return edge_pairs, edge_present


def convex_hull_octahedron_test_gpu(points, eps=1e-9, constants=None):
    """
    Returns a GPU bool tensor:
        shape () for one point set
        shape (B,) for batched point sets
    """
    squeeze = points.ndim == 2
    if squeeze:
        points = points[None, :, :]

    device = points.device
    if constants is None:
        constants = make_hull_constants(device)

    edge_pairs, edge_present = convex_hull_edge_mask_gpu(
        points, eps=eps, constants=constants
    )

    _, _, _, _, incidence = constants

    # Avoid integer matmul on CUDA
    degrees = (edge_present[:, :, None] & incidence[None, :, :]).sum(dim=1)

    edge_count_ok = edge_present.sum(dim=-1) == 12
    degrees_ok = (degrees == 4).all(dim=-1)

    ok = edge_count_ok & degrees_ok

    if squeeze:
        ok = ok[0]

    return ok

def gpu_sim(N, device, rng=torch.Generator(), verbose=False):
    # generate N collections of 6 points (octahedra) on the unit sphere (R^3)
    points_batch = torch.randn(N, 6, 3, generator=rng, device=device, dtype=torch.float64)
    points_batch_norm = points_batch / torch.norm(points_batch, 2, dim=2, keepdim=True)

    constants = make_hull_constants(device)
    with torch.inference_mode():
        ok_batch = convex_hull_octahedron_test_gpu(points_batch_norm, constants=constants)

    x = ok_batch.sum().item()

    if verbose: print(f'successes: {x:,}\ntrials:  {N:,}\np:       {x / N}')
    return x


def main(target=10_000_000_000, reset_temp=78, max_temp=85, verbose=True):
    device = "cuda"

    rng = torch.Generator(device=device)
    seed = rng.seed()
    print(f"{seed=}")

    PATH = r"data"
    N = 500_000
    s = 0
    every = 100
    data = []

    start = time.time()
    wait = 0

    batches = range(0, target, N)
    first = True
    for i, sims in enumerate(batches):
        x = gpu_sim(N, device=device, rng=rng)
        s += x
        data.append(x)
        
        temp = utils.check_temp()
        if temp >= max_temp:
            # Cool off
            while utils.check_temp() >= reset_temp:
                time.sleep(wait_time:=0.25)
                wait += wait_time

        if not first and i % every == 0 or i == (len(batches) - 1):
            pd.DataFrame(
                {"regular": data, 
                 "seed": -1,
                 "N": N
                 }
                 ).to_pickle(f'{PATH}\\gpu_results_{seed}')
            print(f'{i: 9}: {sims=: 16,}, p: {s / sims:.080}, temp: {temp:03}*C, wait: {wait}')
            data = []
            seed = rng.seed()
        else:
            first = False

    print(f"Target reached in {time.time() - start} seconds.\nWaited a total of {wait} seconds to prevent overheating.")


if __name__ == "__main__":
    main(10_000_000_000)