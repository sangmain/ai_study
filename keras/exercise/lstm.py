import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

#reshape into 3 dim
x = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9],[8,9,10],[9,10,11],[10,11,12]])
y = array([4,5,6,7,8,9,10,11,12,13])

print("x.shape: ", x.shape)
print("y.shape: ", y.shape)


x = x.reshape((x.shape[0], x.shape[1], 1))
print("x.shape: ", x.shape)