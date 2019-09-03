from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, LSTM
from keras.callbacks import EarlyStopping

import numpy
import os
import tensorflow as tf

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

print(X_train.shape)
print(X_test.shape)

X_train = X_train.reshape(-1, 28*28, 1).astype('float32') / 255
X_test = X_test.reshape(-1, 28*28, 1).astype('float32') / 255



Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)

# print(X_train.shape)
# print(X_test.shape)

print(Y_train.shape)
print(Y_test.shape)

model = Sequential()
model.add(LSTM(60, input_shape=(28*28,1)))
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test),
                    epochs=1, batch_size=1024, verbose=1, callbacks=[early_stopping_callback])


print("\n Test Accuracy: %.4f" % (model.evaluate(X_test, Y_test)[1]))


