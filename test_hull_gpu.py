import numpy as np
import pytest
import torch

from octahedral_gpu import (
    make_hull_constants,
    convex_hull_edge_mask_gpu,
    convex_hull_octahedron_test_gpu,
)
from octahedral import (
    convex_hull_edges,
    convex_hull_octahedron_test,
)


def gpu_edges_as_set(points_np, constants=None, eps=1e-9):
    """
    Test helper: converts GPU edge mask back into a Python set
    so we can compare against the CPU implementation.
    """
    points_t = torch.as_tensor(points_np, device="cuda", dtype=torch.float64)

    if constants is None:
        constants = make_hull_constants(points_t.device)

    edge_pairs, edge_present = convex_hull_edge_mask_gpu(
        points_t,
        eps=eps,
        constants=constants,
    )

    edge_pairs_cpu = edge_pairs[edge_present].detach().cpu().numpy()

    return {
        tuple(sorted(map(int, edge)))
        for edge in edge_pairs_cpu
    }


def gpu_bool(points_np, constants=None, eps=1e-9):
    points_t = torch.as_tensor(points_np, device="cuda", dtype=torch.float64)

    if constants is None:
        constants = make_hull_constants(points_t.device)

    result = convex_hull_octahedron_test_gpu(
        points_t,
        eps=eps,
        constants=constants,
    )

    return bool(result.detach().cpu().item())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_regular_octahedron_edges_agree():
    points = np.array([
        [ 1.0,  0.0,  0.0],
        [-1.0,  0.0,  0.0],
        [ 0.0,  1.0,  0.0],
        [ 0.0, -1.0,  0.0],
        [ 0.0,  0.0,  1.0],
        [ 0.0,  0.0, -1.0],
    ], dtype=np.float64)

    constants = make_hull_constants("cuda")

    cpu_edges = convex_hull_edges(points)
    gpu_edges = gpu_edges_as_set(points, constants=constants)

    assert gpu_edges == cpu_edges


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_regular_octahedron_boolean_agrees():
    points = np.array([
        [ 1.0,  0.0,  0.0],
        [-1.0,  0.0,  0.0],
        [ 0.0,  1.0,  0.0],
        [ 0.0, -1.0,  0.0],
        [ 0.0,  0.0,  1.0],
        [ 0.0,  0.0, -1.0],
    ], dtype=np.float64)

    constants = make_hull_constants("cuda")

    assert convex_hull_octahedron_test(points) is True
    assert gpu_bool(points, constants=constants) is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_affine_transformed_octahedron_agrees():
    """
    A nonsingular affine transform preserves the combinatorial hull graph.
    So this should still be octahedral.
    """
    points = np.array([
        [ 1.0,  0.0,  0.0],
        [-1.0,  0.0,  0.0],
        [ 0.0,  1.0,  0.0],
        [ 0.0, -1.0,  0.0],
        [ 0.0,  0.0,  1.0],
        [ 0.0,  0.0, -1.0],
    ], dtype=np.float64)

    A = np.array([
        [2.0, 0.3, 0.1],
        [0.2, 1.5, 0.4],
        [0.1, 0.2, 1.2],
    ], dtype=np.float64)

    b = np.array([10.0, -3.0, 5.0], dtype=np.float64)

    transformed = points @ A.T + b

    constants = make_hull_constants("cuda")

    assert gpu_edges_as_set(transformed, constants=constants) == convex_hull_edges(transformed)
    assert gpu_bool(transformed, constants=constants) == convex_hull_octahedron_test(transformed)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_non_octahedron_agrees():
    """
    Six points where one point is clearly inside a tetrahedron-like hull.
    This should not be octahedral.
    """
    points = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
        [0.25, 0.25, 0.25],
        [0.10, 0.20, 0.15],
    ], dtype=np.float64)

    constants = make_hull_constants("cuda")

    assert gpu_edges_as_set(points, constants=constants) == convex_hull_edges(points)
    assert gpu_bool(points, constants=constants) == convex_hull_octahedron_test(points)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_random_general_position_edges_agree():
    """
    Random points are almost surely nondegenerate, so the CPU and GPU edge
    extraction should agree.
    """
    rng = np.random.default_rng(12345)
    constants = make_hull_constants("cuda")

    for _ in range(200):
        points = rng.normal(size=(6, 3)).astype(np.float64)

        cpu_edges = convex_hull_edges(points)
        gpu_edges = gpu_edges_as_set(points, constants=constants)

        assert gpu_edges == cpu_edges


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_random_general_position_boolean_agrees():
    rng = np.random.default_rng(54321)
    constants = make_hull_constants("cuda")

    for _ in range(200):
        points = rng.normal(size=(6, 3)).astype(np.float64)

        cpu_result = convex_hull_octahedron_test(points)
        gpu_result = gpu_bool(points, constants=constants)

        assert gpu_result == cpu_result


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_batched_gpu_matches_cpu_loop():
    rng = np.random.default_rng(2468)

    batch_np = rng.normal(size=(100, 6, 3)).astype(np.float64)
    batch_t = torch.as_tensor(batch_np, device="cuda", dtype=torch.float64)

    constants = make_hull_constants("cuda")

    gpu_result = convex_hull_octahedron_test_gpu(
        batch_t,
        constants=constants,
    )

    gpu_result_np = gpu_result.detach().cpu().numpy()

    cpu_result_np = np.array([
        convex_hull_octahedron_test(points)
        for points in batch_np
    ])

    np.testing.assert_array_equal(gpu_result_np, cpu_result_np)