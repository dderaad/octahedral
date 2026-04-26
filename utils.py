import numpy as np

def sample_points_on_sphere(rng: np.random.Generator, n=6):
    x = rng.normal(size=(n, 3))
    x /= np.linalg.norm(x, axis=1)[:, None]
    return x