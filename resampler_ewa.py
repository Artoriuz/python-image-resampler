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
def mitchell(x, a):
    x = abs(x)
    c = 0.5
    b = 0
    if x <= 1:
        return (1/6)*((12 - 9*b - 6*c)*(x**3) + (-18 + 12*b + 6*c)*(x**2) + (6 - 2*b))
    elif 1 < x <= 2:
        return (1/6)*((-b - 6*c)*(x**3) + (6*b + 30*c)*(x**2) + (-12*b - 48*c)*x + (8*b + 24*c))
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
        return (25/16 * (2/5 * x**2 - 1)**2 - (25/16 - 1)) * (1/4 * x**2 - 1)**2
    else:
        return 0

@njit
def ewa_lanczos2_polynomial(x):
    x = abs(x)
    if x < 2.2331305943815286:
        return -0.35398 * x**4 + 1.73871 * x**3 + -2.39193 * x**2 + 0.15839 * x + 0.99584
    else:
        return 0

@njit
def ewa_lanczos3_polynomial(x):
    x = abs(x)
    if x < 3.2383154841662362:
        return 0.08614 * x**5 + -0.79902 * x**4 + 2.58326 * x**3 + -3.14569 * x**2 + 0.45981 * x + 0.97632
    else:
        return 0

@njit
def dist(x, y):
    return np.sqrt(np.square(x) + np.square(y))

@njit
def elliptical_resampler(input, factor):
    input_shape = input.shape
    output_shape = (int(round(input_shape[0] * factor)), int(round(input_shape[1] * factor)), input_shape[2])
    output = np.empty(output_shape, dtype=np.float32)

    filter_size = 3
    anti_ringing = True

    for y in range(output_shape[0]):
        for x in range(output_shape[1]):
            for z in range(output_shape[2]):
                equiv_y = float(y) / factor
                equiv_x = float(x) / factor

                idy_start = np.maximum(np.floor(equiv_y) - np.ceil(filter_size) + 1, 0.0)
                idy_end = np.minimum(np.floor(equiv_y) + np.ceil(filter_size) + 1, input_shape[0])
                idx_start = np.maximum(np.floor(equiv_x) - np.ceil(filter_size) + 1, 0.0)
                idx_end = np.minimum(np.floor(equiv_x) + np.ceil(filter_size) + 1, input_shape[1])

                norm = 0.0
                val = 0.0

                if anti_ringing:
                    inputs_for_clamping = []

                for idy in range(idy_start, idy_end):
                    for idx in range(idx_start, idx_end):
                        # weight = catmull_rom(dist(equiv_x - idx, equiv_y - idy))
                        # weight = amd_lanczos(dist(idx - equiv_x, idy - equiv_y))
                        # weight = lanczos_2d(idx - equiv_x, idy - equiv_y, filter_size)
                        # weight = catrom_2d(idx - equiv_x, idy - equiv_y)
                        weight = ewa_lanczos2_polynomial(dist(equiv_x - idx, equiv_y - idy))
                        norm += weight
                        val += input[idy, idx, z] * weight
                        if anti_ringing:
                            inputs_for_clamping.append(input[idy, idx, z])

                if norm == 0:
                    norm = 0.01

                if anti_ringing:
                    output[y, x, z] = np.minimum(np.amax(np.array(inputs_for_clamping)), np.maximum(val / norm, np.amin(np.array(inputs_for_clamping))))
                else:
                    output[y, x, z] = val / norm

    return output

array = (cv2.imread("input.png") / 255.0).astype(np.float64)
test = elliptical_resampler(array, 2.0)

test = np.clip(test * 255.0, 0, 255).astype(np.uint8)
cv2.imwrite("ewa_output.png", test)
