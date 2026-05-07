import numpy as np
import pynvml

def sample_points_on_sphere(rng: np.random.Generator, n=6):
    x = rng.normal(size=(n, 3))
    x /= np.linalg.norm(x, axis=1)[:, None]
    return x

def first_n_digits(s1, s2):
    if type(s1) is not str: s1 = str(s1)
    if type(s2) is not str: s2 = str(s2)

    for i, chars in enumerate(zip(s1, s2)):
        if chars[0] == chars[1]:
            continue
        return i
    else:
        return 0

def check_temp():
    pynvml.nvmlInit()

    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    temp_c = pynvml.nvmlDeviceGetTemperature(
        handle,
        pynvml.NVML_TEMPERATURE_GPU,
    )

    return temp_c