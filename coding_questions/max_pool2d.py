import numpy as np
x = np.arange(1, 17).reshape(1, 1, 4, 4)

def overlapping_max_pool2d(input_matrix, kernel_size=2, stride=2):

    batch_size, channels, input_height, input_width = input_matrix.shape

    output_height = ((input_height - 1) // stride) + 1
    output_width = ((input_width - 1) // stride) + 1

    output_matrix = np.zeros((batch_size, channels, output_height, output_width))
    for b in range(batch_size):
        for c in range(channels):
            for i in range(0, output_height):
                for j in range(0, output_width):
                    start_height = i * stride
                    start_width = j * stride
                    end_height = min(start_height + kernel_size, input_height)
                    end_width = min(start_width + kernel_size, input_width)
                    output_matrix[b, c, i, j] = np.max(input_matrix[b, c, start_height:end_height, start_width:end_width])

    return output_matrix
print(overlapping_max_pool2d(x, kernel_size=3, stride=2))