from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense, Activation
from keras import backend as K


class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()

        input_shape = (width, height, depth)
        if K.image_data_format() == "channels_first":
            input_shape = (depth, width, height)

        model.add(Conv2D(32, (3, 3), padding="same", input_shape=input_shape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(50, (5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Flatten())
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model
