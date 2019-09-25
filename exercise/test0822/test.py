import numpy as np

x_train = [
    [1,2,3,4,5,None, None, None, 9, 10, 11],
    [2,3,4,5,6, None, None, None, 10, 11, 12],
    [50,51,52,53,54, None, None, None, 58, 59, 60]]
y_train = [[6, 7, 8], [7,8,9], [55, 56, 57]]

x_test = [[35,36,37,38,39, None, None, None, 43, 44, 45]]

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)

x_train = x_train.reshape(-1, 11, 1)
x_test = x_test.reshape(-1, 11, 1)

from keras.models import Sequential
from keras.layers import LSTM, Dense

model = Sequential()

model.add(LSTM(50, input_shape = (11,1), activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(512, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
model.add(Dense(3, activation='relu'))

model.compile(optimizer='adam', loss='mse', metrics=['mse'])
model.fit(x_train, y_train, epochs=30, batch_size=1, verbose=2)


pred = model.predict(x_test, batch_size=1)
print(pred)