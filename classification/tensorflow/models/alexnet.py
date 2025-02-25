""" Alexnet implementation model
"""
from .model import BaseModel
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout

class AlexNetFromScratch(BaseModel):
    def build(self):
        loss, activation = ("binary_crossentropy", "sigmoid") if self.classes == 2 else ("categorical_crossentropy", "softmax")
        model = Sequential(
            [
                Input(shape=self._input_shape),
                Conv2D(96, (11, 11), strides=4, activation='relu', input_shape=self._input_shape),
                MaxPooling2D(pool_size=(3, 3), strides=2),
                Conv2D(256, (5, 5), activation='relu', padding="same"),
                MaxPooling2D(pool_size=(3, 3), strides=2),
                Conv2D(384, (3, 3), activation='relu', padding="same"),
                Conv2D(384, (3, 3), activation='relu', padding="same"),
                Conv2D(256, (3, 3), activation='relu', padding="same"),
                MaxPooling2D(pool_size=(3, 3), strides=2),
                Flatten(),
                Dense(4096, activation='relu'),
                Dropout(0.5),
                Dense(4096, activation='relu'),
                Dropout(0.5),
                Dense(self._num_classes, activation=activation)
            ]
        )
        model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
        return model