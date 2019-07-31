import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, BatchNormalization

a = np.array(range(1,101))
batch_size = 1
size = 5

def split_5(seq, size):
    aaa = []
    for i in range(len(a)-size+1):
        subset = a[i:(i+size)]
        aaa.append(subset)

    # print(type(aaa))
    return np.array(aaa)


dataset = split_5(a, size)
print('=================')
# print(dataset)
# print(dataset.shape)

x_train = dataset[:,0:4]
y_train = dataset[:, 4]

x_train = np.reshape(x_train,(len(x_train), size-1, 1))

x_test = x_train + 100
y_test = y_train + 100

# print(x_train.shape)
# print(y_train.shape)

# print(x_test)
# print(y_test)


model = Sequential()
model.add(LSTM(128, batch_input_shape=(1,4,1), stateful=True))
# model.add(Dropout(0.5))


# model.add(Dense(512))
model.add(Dense(50, activation='relu'))

model.add(Dense(12, activation='relu'))


model.add(Dense(15, activation='relu'))
# model.add(Dense(6, activation='relu'))

model.add(Dense(1))
model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

from keras.callbacks import EarlyStopping

early = EarlyStopping(monitor='mse', patience=30)

num_epochs = 200

loss = []
valloss = []

for epochs_idx in range(num_epochs):
    print('epochs:' + str(epochs_idx))
    history = model.fit(x_train, y_train, epochs = 1, batch_size=batch_size, verbose=2, shuffle= False, validation_data=(x_test, y_test), callbacks=[early])
    
    loss.append(history.history['mean_squared_error'])
    valloss.append(history.history['val_mean_squared_error'])

    model.reset_states()

mse, _ = model.evaluate(x_train, y_train, batch_size)
print('mse: ', mse)
model.reset_states()

y_predict = model.predict(x_test, batch_size=1)

print(y_predict[0:5])

from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE: ", RMSE(y_test, y_predict))

#R2 구하기
from sklearn.metrics import r2_score

r2_y_predict = r2_score(y_test, y_predict)
print("R2: ", r2_y_predict)

import matplotlib.pyplot as plt


# for x in history:


# print(loss)
# # print
plt.plot(loss)
plt.plot(valloss)

plt.title('model loss')
plt.ylabel('mse')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show() 
