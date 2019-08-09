import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt

data = pd.read_csv("kospi200test_r.csv")

# 최근 20일의 데이터만 갖고온다
# train = data[-20:]
train = data[:]
end_prices = train['종가']

# print(train['일자'])


# #normalize window
seq_len = 5
sequence_length = seq_len + 1

result = []
for index in range(len(end_prices) - sequence_length + 1):
    idk = []
    idk[:] = end_prices[index: index + sequence_length]
    result.append(idk)


result = np.array(result)
# np.random.shuffle(result)



x = result[:, :-1]
y = result[:, -1]

def normalize_data(data):
    normalized_data = []
    for x in data:
        normalized_window = [((float(p) / float(x[0])) - 1) for p in x]
        normalized_data.append(normalized_window)
    return np.array(normalized_data)

# norm_x = normalize_data(x)
norm_x = x

split_ratio = int(len(norm_x) * 0.9)

x_train = norm_x[:split_ratio, :]
x_test = norm_x[split_ratio:, :]

y_train = y[:split_ratio]
y_test = y[split_ratio:]

# print(x_train[59])
# print(y_train[59])

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

print(x_train.shape)
print(x_test.shape)

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation

model = Sequential()

model.add(LSTM(50, return_sequences=True, input_shape=(5, 1)))

model.add(LSTM(64, return_sequences=False))

model.add(Dense(1, activation='linear'))

model.compile(loss='mse', optimizer='rmsprop')

model.summary()

model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=1, epochs=20)

model.save("model.h5")