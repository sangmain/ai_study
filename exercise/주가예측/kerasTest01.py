import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt

train = pd.read_csv("cheatkey.csv")

end_prices = train['종가']
print(type(end_prices))
#normalize window
seq_len = 50
sequence_length = seq_len + 1

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

norm_result = normalize_windows(result)

# split train and test data
row = int(round(norm_result.shape[0] * 0.9))
train = norm_result[:row, :]
np.random.shuffle(train)

x_train = train[:, :-1]
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
y_train = train[:, -1]

x_test = norm_result[row:, :-1]
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
y_test = norm_result[row:, -1]

print(x_test.shape)
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation

model = Sequential()

model.add(LSTM(50, return_sequences=True, input_shape=(seq_len, 1)))

model.add(LSTM(64, return_sequences=False))

model.add(Dense(1, activation='linear'))

model.compile(loss='mse', optimizer='rmsprop')

model.summary()

model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=int(seq_len/5), epochs=20)
model.save("model.h5")

pred = model.predict(x_test)

result = []
result[:] = end_prices[-seq_len:]

result = np.array(result)
x_test = result.reshape(1, -1, 1)

print('역정규화 개시')
un_norm = result[0]
pred_today = (pred[-1]+1) * un_norm

print("last 5 days:\n ", x_test)
print("prediction: ", pred_today)


fig = plt.figure(facecolor='white')
ax = fig.add_subplot(111)
ax.plot(y_test, label='True')
ax.plot(pred, label='Prediction')

ax.legend()
plt.show()