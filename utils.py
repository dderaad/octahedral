import numpy as np

def sample_points_on_sphere(n=6):
    x = np.random.normal(size=(n, 3))
    x /= np.linalg.norm(x, axis=1)[:, None]
    return x