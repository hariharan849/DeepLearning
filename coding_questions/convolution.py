import numpy as np

input_matrix = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])

kernel = np.array([
    [1, 0],
    [-1, 1]
])

padding = 1
stride = 2

def simple_conv2d(input_matrix, kernel, padding=0, stride=1):
    input_height, input_width = input_matrix.shape
    kernel_height, kernel_width = kernel.shape

    padded_input = np.pad(input_matrix, mode='constant', constant_values=0, pad_width=padding)

    output_height = ((input_height + (2 * padding) - kernel_height) // stride) + 1
    output_width = ((input_width + (2 * padding) - kernel_width) // stride) + 1

    output_matrix = np.zeros((output_height, output_width))

    for i in range(0, output_height):
        for j in range(0, output_width):
            start_height = i * stride
            start_width = j * stride
            output_matrix[i, j] = np.sum(padded_input[start_height:start_height + kernel_height, start_width:start_width + kernel_width] * kernel)

    return output_matrix
output = simple_conv2d(input_matrix, kernel, padding, stride)
print(output)
