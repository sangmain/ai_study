import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt

train = pd.read_csv("kospi200test_r.csv")

end_prices = train['종가']

#normalize window
x_len = 40
y_len = 10
sequence_length = x_len + y_len

result = []
for index in range(len(end_prices) - sequence_length + 1):
    idk = []
    idk[:] = end_prices[index: index + sequence_length]
    result.append(idk)

#normalize data
def normalize_windows(data):
    normalized_data = []
    for window in data:
        normalized_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalized_data.append(normalized_window)
    return np.array(normalized_data)

# norm_result = normalize_windows(result)
norm_result = np.array(result)

# split train and test data
row = int(round(norm_result.shape[0] * 0.9))
train = norm_result[:row, :]
np.random.shuffle(train)

x_train = train[:, : -y_len]
# x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
y_train = train[:, -y_len: ]


x_test = norm_result[row:, :-y_len]
# x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
y_test = norm_result[row:, -y_len: ]

print(x_test.shape)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

print(x_test[0])
print(y_test[0])
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation, Conv1D, Embedding, MaxPooling1D

model = Sequential()
model.add(Embedding(5000, 100))

model.add(Dropout(0.5))

model.add(Conv1D(64, 5, padding='valid', activation='relu', strides=1))

model.add(MaxPooling1D(pool_size=4))

model.add(LSTM(55))

model.add(Dense(10))

model.add(Activation('linear'))
model.compile(loss='mse', optimizer='rmsprop')

model.summary()

model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=int(x_len/5), epochs=20)
model.save("model2.h5")

pred = model.predict(x_test)

fig = plt.figure(facecolor='white')
ax = fig.add_subplot(111)
ax.plot(y_test, label='True')
ax.plot(pred, label='Prediction')

ax.legend()
plt.show()

# result = []
# result[:] = end_prices[-seq_len:]

# result = np.array(result)
# x_test = result.reshape(1, -1, 1)

# print('역정규화 개시')
# un_norm = result[0]
# pred_today = (pred[-1]+1) * un_norm

# print("last 5 days:\n ", x_test)
# print("prediction: ", pred_today)


