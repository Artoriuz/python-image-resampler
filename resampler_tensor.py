import numpy as np
import cv2
from numba import njit

@njit
def lanczos(x, a):
    return np.sinc(x) * np.sinc(x / a)

@njit
def signal_resampler(input, factor):
    input_shape = input.shape
    output_shape = int(round(input_shape[0] * factor))

    output = np.zeros(output_shape)
    filter_size = 3

    for x in range(output_shape):
        equiv_x = x / factor
        idx_start = np.maximum(np.floor(equiv_x) - filter_size, 0.0)
        idx_end = np.minimum(np.floor(equiv_x) + filter_size + 1, input_shape[0])
        norm = 0.0
        val = 0.0
        for idx in range(idx_start, idx_end):
            weight = lanczos(equiv_x - idx, filter_size) 
            norm += weight
            val += input[idx] * weight
        output[x] = val / norm

    return output

@njit
def image_resampler(input, factor):
    input_shape = input.shape
    output_shape = (int(round(input_shape[0] * factor)), int(round(input_shape[1] * factor)), input_shape[2])
    intermediate = np.empty((input_shape[0], output_shape[1], input_shape[2]), dtype=np.float32)
    output = np.empty(output_shape, dtype=np.float32)
    for row in range(input_shape[0]):
        for channel in range(input_shape[2]):
            intermediate[row, :, channel] = signal_resampler(input[row, :, channel], factor)
    
    for col in range(output_shape[1]):
        for channel in range(input_shape[2]):
            output[:, col, channel] = signal_resampler(intermediate[:, col, channel], factor)
    
    return output

array = (cv2.imread(f"input.png") / 255.0).astype(np.float32)
test = image_resampler(array, 2)

test = np.clip(test * 255.0, 0, 255).astype(np.uint8)
cv2.imwrite("output.png", test)
