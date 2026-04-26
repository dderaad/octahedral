import numpy as np
from scipy.spatial import ConvexHull
from collections import Counter
from itertools import combinations

"""
Credit: ViliamF
"""

def convex_hull_edges(points, eps=1e-9):
    """
    Compute edges of the convex hull for 6 points in 3D.

    Parameters:
        points: np.ndarray of shape (6, 3), dtype=float64
        eps: tolerance for numerical stability

    Returns:
        edges: set of (i, j) index pairs (i < j)
    """
    n = points.shape[0]
    assert n == 6 and points.shape[1] == 3
    
    faces = []
    
    # Check all triangles
    for i, j, k in combinations(range(n), 3):
        p0, p1, p2 = points[i], points[j], points[k]
        
        # Compute normal vector of triangle
        normal = np.cross(p1 - p0, p2 - p0)
        
        # Skip degenerate triangles
        if np.linalg.norm(normal) < eps:
            continue
        
        # Determine side of all other points
        signs = []
        for m in range(n):
            if m in (i, j, k):
                continue
            d = np.dot(normal, points[m] - p0)
            if abs(d) < eps:
                signs.append(0)
            elif d > 0:
                signs.append(1)
            else:
                signs.append(-1)
        
        # Check if all points lie on one side
        nonzero = [s for s in signs if s != 0]
        if len(nonzero) == 0 or all(s == nonzero[0] for s in nonzero):
            faces.append((i, j, k))
    
    # Extract unique edges from faces
    edges = set()
    for i, j, k in faces:
        for a, b in [(i, j), (j, k), (k, i)]:
            edges.add(tuple(sorted((a, b))))
    
    return edges


def convex_hull_octahedron_test(points):
    edges = convex_hull_edges(points)
    
    degree = Counter()
    for a, b in edges:
        degree[a] += 1
        degree[b] += 1
    
    degrees_ok = all(d == 4 for d in degree.values())
    
    return degrees_ok