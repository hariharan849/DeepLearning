from numpy import asarray
from keras.models import Sequential
from keras.layers import Dense, UpSampling2D, Reshape, Conv2D, Conv2DTranspose


model = Sequential(
    [
        Conv2DTranspose(1, (1,1), strides=(2,2), input_shape=(2, 2, 1))
    ]
)
weights = [asarray([[[[1]]]]), asarray([0])]
# store the weights in the model
model.set_weights(weights)

model.summary()
x = asarray([[1, 2], [3, 4]])
print(x)
x = x.reshape((1, 2, 2, 1))
print(x)

yhat = model.predict(x)
print(yhat.reshape((4, 4)))