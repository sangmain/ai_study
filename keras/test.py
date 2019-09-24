from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

import numpy
import numpy as np

import os
import tensorflow as tf

(X_train_all, Y_train_all), (X_test, Y_test) = mnist.load_data()
import matplotlib.pyplot as plt

print(X_train_all.shape)
print()

from sklearn.model_selection import train_test_split
X_train, __, Y_train, __ = train_test_split(X_train_all, Y_train_all, random_state = 66, test_size = 0.995)
# X_train = X_train_all[:300]
# Y_train = Y_train_all[:300]
print(X_train.shape)
print(Y_train.shape)

from keras.preprocessing.image import ImageDataGenerator

data_generator = ImageDataGenerator(rotation_range=20,
                                width_shift_range = 0.02,
                                height_shift_range=0.02,
                                horizontal_flip=True
                            )



X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255

Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)

# print(X_train.shape)
# print(X_test.shape)

model = Sequential()
model.add(Conv2D(16, kernel_size=(6,6), input_shape=(28,28,1), activation='relu'))
model.add(Conv2D(8,(12,12), activation='relu'))
# model.add(Conv2D(128,(3,3), activation='relu'))
# model.add(Conv2D(256,(3,3), activation='relu'))


model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax')) #분류모델의 마지막은 softmax

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# early_stopping_callback = EarlyStopping(monitor='loss', patience=10)

model.fit_generator(data_generator.flow(X_train, Y_train, batch_size=64), steps_per_epoch= len(X_train) // 32, epochs = 200, validation_data = (X_test, Y_test), verbose=1)#, callbacks=[early_stopping_callback])


print("\n Test Accuracy: %.4f" % (model.evaluate(X_test, Y_test)[1]))


