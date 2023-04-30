import numpy as np
import cv2
from numba import njit

@njit
def j1(x):
    # For small values of x, use Taylor series expansion
    if abs(x) < 1e-8:
        return x / 3.0 - (x**3) / 30.0 + (x**5) / 840.0 - (x**7) / 45360.0 + (x**9) / 3991680.0

    # For large values of x, use asymptotic expansion
    if abs(x) > 3.0:
        return (np.sin(x) / x - np.cos(x)) / np.sqrt(x)

    # For intermediate values of x, use rational approximation
    x2 = x**2
    return x * (0.5 + x2 * (-0.0187293 + x2 * (0.000595238 + x2 * (-8.33333e-6 + x2 * 8.2672e-8))))

@njit
def jinc(x):
    if abs(x) < 1e-8:
        return 1.0
    x *= np.pi
    return 2.0 * j1(x) / x

@njit
def ewa_lanczos(x, a):
    return jinc(x) * jinc(x / a)

@njit
def dist(x, y):
    return np.sqrt(np.square(x) + np.square(y))

@njit
def elliptical_resampler(input, factor):
    input_shape = input.shape
    output_shape = (int(round(input_shape[0] * factor)), int(round(input_shape[1] * factor)), input_shape[2])
    output = np.empty(output_shape, dtype=np.float32)

    filter_size = 3
    
    for y in range(output_shape[0]):
        for x in range(output_shape[1]):
            for z in range(output_shape[2]):
                equiv_y = y / factor
                equiv_x = x / factor

                idy_start = np.maximum(np.floor(equiv_y) - filter_size, 0.0)
                idy_end = np.minimum(np.floor(equiv_y) + filter_size + 1, input_shape[0])
                idx_start = np.maximum(np.floor(equiv_x) - filter_size, 0.0)
                idx_end = np.minimum(np.floor(equiv_x) + filter_size + 1, input_shape[1])

                norm = 0.0
                val = 0.0

                for idy in range(idy_start, idy_end):
                    for idx in range(idx_start, idx_end):
                        weight = ewa_lanczos(dist(equiv_x - idx, equiv_y - idy), filter_size)
                        norm += weight
                        val += input[idy, idx, z] * weight
                if norm == 0:
                    norm = 0.001
                output[y, x, z] = val / norm

    return output

array = (cv2.imread("input.png") / 255.0).astype(np.float32)
test = elliptical_resampler(array, 2.0)

test = np.clip(test * 255.0, 0, 255).astype(np.uint8)
cv2.imwrite("ewa_output.png", test)
