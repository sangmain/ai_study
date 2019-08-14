import numpy as np

(x_train, y_train), (x_test, y_test) = np.load("cifar.npy")

x_train = x_train[:2000]
y_train = y_train[:2000]

x_test = x_test[:2000]
y_test = y_test[:2000]
print(x_train.shape, x_test.shape)

from keras.preprocessing.image import img_to_array, array_to_img
x_train = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48,48))) for im in x_train])
x_test = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48,48))) for im in x_test])

print(x_train.shape, x_test.shape)

from keras.utils import to_categorical

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


from keras.applications import VGG16
conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(48,48,3))

from keras import models
from keras.models import Model, Sequential
from keras.layers import Flatten, Dense
from keras import optimizers
from keras import callbacks
from keras.layers.advanced_activations import LeakyReLU


model = Sequential()

model.add(conv_base)
model.add(Flatten())
model.add(Dense(256))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss="categorical_crossentropy", optimizer="adam",metrics=["acc"])
model.fit(x_train,y_train,epochs=100, batch_size=500)

print("acc: ",model.evaluate(x_test,y_test)[1])