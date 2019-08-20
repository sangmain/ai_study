import tensorflow as tf
tf.set_random_seed(777)
import numpy as np
import pandas as pd

data = np.genfromtxt("./data/data-04-zoo.csv", delimiter=',')
x_data = data[:-1, 0:-1]
y_data = data[:-1, [-1]]

print(x_data.shape, y_data.shape)
x_test = data[-10:, 0:-1]
y_test = data[-10:, [-1]]


print(x_test.shape, y_test.shape)

from keras.utils import to_categorical
y_data = to_categorical(y_data)
y_test = to_categorical(y_test)

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(7, input_shape=(16,), activation='softmax'))


