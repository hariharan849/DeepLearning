# AlexNet Architecture

AlexNet is a convolutional neural network that is 8 layers deep. Here is a summary of the architecture:

1. **Input Layer**: 224x224x3 image
2. **Conv Layer 1**: 96 filters of size 11x11, stride 4, padding 0
3. **Max Pooling Layer 1**: 3x3 filters, stride 2
4. **Conv Layer 2**: 256 filters of size 5x5, stride 1, padding 2
5. **Max Pooling Layer 2**: 3x3 filters, stride 2
6. **Conv Layer 3**: 384 filters of size 3x3, stride 1, padding 1
7. **Conv Layer 4**: 384 filters of size 3x3, stride 1, padding 1
8. **Conv Layer 5**: 256 filters of size 3x3, stride 1, padding 1
9. **Max Pooling Layer 3**: 3x3 filters, stride 2
10. **Fully Connected Layer 1**: 4096 neurons
11. **Fully Connected Layer 2**: 4096 neurons
12. **Fully Connected Layer 3**: 1000 neurons (output layer with softmax activation)


![AlexNet Architecture](https://upload.wikimedia.org/wikipedia/commons/thumb/c/cc/Comparison_image_neural_networks.svg/480px-Comparison_image_neural_networks.svg.png)



## References

- Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks. In Advances in neural information processing systems (pp. 1097-1105).


# ResNet Architecture

ResNet (Residual Network) is a convolutional neural network that introduces residual learning to ease the training of networks that are substantially deeper than those used previously. Here is a summary of the architecture for ResNet-50:

1. **Input Layer**: 224x224x3 image
2. **Conv Layer 1**: 64 filters of size 7x7, stride 2, padding 3
3. **Max Pooling Layer 1**: 3x3 filters, stride 2
4. **Conv Block 1**: 3 layers of 64 filters of size 1x1, 3x3, 1x1
5. **Identity Block 1**: 3 layers of 64 filters of size 1x1, 3x3, 1x1
6. **Conv Block 2**: 3 layers of 128 filters of size 1x1, 3x3, 1x1
7. **Identity Block 2**: 3 layers of 128 filters of size 1x1, 3x3, 1x1
8. **Conv Block 3**: 3 layers of 256 filters of size 1x1, 3x3, 1x1
9. **Identity Block 3**: 3 layers of 256 filters of size 1x1, 3x3, 1x1
10. **Conv Block 4**: 3 layers of 512 filters of size 1x1, 3x3, 1x1
11. **Identity Block 4**: 3 layers of 512 filters of size 1x1, 3x3, 1x1
12. **Average Pooling Layer**: 7x7 filters
13. **Fully Connected Layer**: 1000 neurons (output layer with softmax activation)

![ResNet Architecture](https://cdn.analyticsvidhya.com/wp-content/uploads/2021/08/1_rOFPhrpfwguotGdB1-BseA.webp)

## References

- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).