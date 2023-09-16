import numpy as np
import cv2
from numba import njit

@njit
def catmull_rom(x):
    x = abs(x)
    if x <= 1.0:
        return (1.5*x**3 - 2.5*x**2 + 1)
    elif 1.0 < abs(x) <= 2.0:
        return (-0.5*x**3 + 2.5*x**2 - 4*x + 2)
    else:
        return 0.0

@njit
def catrom_2d(x, y):
    return catmull_rom(x) * catmull_rom(y)

@njit
def lanczos(x, a):
    if abs(x) < a:
        return np.sinc(x) * np.sinc(x / a)
    else:
        return 0

@njit
def lanczos_2d(x, y, a):
    return lanczos(x, a) * lanczos(y, a)

@njit
def amd_lanczos(x):
    x = abs(x)
    if x < 2:
        return (25.0/16.0 * (2.0/5.0 * x**2.0 - 1.0)**2 - (25.0/16.0 - 1.0)) * (1.0/4.0 * x**2.0 - 1.0)**2.0
    else:
        return 0

@njit
def dist(x, y):
    return np.sqrt(np.square(x) + np.square(y))

@njit
def polar_resampler(input, factor):
    input_shape = input.shape
    output_shape = (int(round(input_shape[0] * factor)), int(round(input_shape[1] * factor)), input_shape[2])
    output = np.empty(output_shape, dtype=np.float32)

    filter_radius = 2
    anti_ringing = True

    for y in range(output_shape[0]):
        for x in range(output_shape[1]):
            for z in range(output_shape[2]):
                equiv_y = (float(y) - 0.5) / factor
                equiv_x = (float(x) - 0.5) / factor

                idy_start = np.maximum(np.floor(equiv_y) - np.ceil(filter_radius), 0.0)
                idy_end = np.minimum(np.floor(equiv_y) + np.ceil(filter_radius) + 1, input_shape[0])
                idx_start = np.maximum(np.floor(equiv_x) - np.ceil(filter_radius), 0.0)
                idx_end = np.minimum(np.floor(equiv_x) + np.ceil(filter_radius) + 1, input_shape[1])

                norm = 0.0
                val = 0.0
                if anti_ringing:
                    inputs_for_clamping = []

                for idy in range(idy_start, idy_end):
                    for idx in range(idx_start, idx_end):
                        # weight = catmull_rom(dist(equiv_x - idx, equiv_y - idy))
                        # weight = amd_lanczos(dist(idx - equiv_x, idy - equiv_y))
                        # weight = lanczos_2d(idx - equiv_x, idy - equiv_y, filter_radius)
                        # weight = catrom_2d(idx - equiv_x, idy - equiv_y)
                        weight = amd_lanczos(dist(equiv_x - idx, equiv_y - idy))
                        norm += weight
                        val += input[idy, idx, z] * weight
                        if anti_ringing:
                            inputs_for_clamping.append(input[idy, idx, z])

                norm = np.maximum(norm, 0.001)

                if anti_ringing:
                    output[y, x, z] = np.minimum(np.amax(np.array(inputs_for_clamping)), np.maximum(val / norm, np.amin(np.array(inputs_for_clamping))))
                else:
                    output[y, x, z] = val / norm

    return output

array = (cv2.imread("input.png") / 255.0).astype(np.float64)
test = polar_resampler(array, 2.0)

test = np.clip(test * 255.0, 0, 255).astype(np.uint8)
cv2.imwrite("polar_output.png", test)
