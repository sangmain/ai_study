from keras.layers import Dense, Dropout, LSTM
import numpy as np

idxx2char = ['e', 'h', 'i', 'l', 'o']

_data = np.array([['h', 'i', 'h', 'e', 'l', 'l', 'o']], dtype=np.str).reshape(-1, 1)


from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
enc.fit(_data)
_data = enc.transform(_data).toarray().astype('float32')

x_data = _data[:6,] 
y_data = _data[1:,]
y_data = np.argmax(y_data, axis=1)
print(y_data)

x_data = x_data.reshape(1, 6, 5)
y_data = y_data.reshape(1, 6)

print(x_data.shape)

num_classes = 5
batch_size = 1
sequence_length = 6
input_dim = 5
hidden_size = 5
learning_rate = 0.1


# def split_5(seq, size):
#     aaa = []
#     for i in range(len(seq)-size+1):
#         subset = seq[i:(i+size)]
#         aaa.append(subset)

#     print(type(aaa))
#     return np.array(aaa)


# x_train = split_5(x_data, 5)
# print(x_train.shape)

# x_data = x_data.reshape(-1, 5, 1)
from keras.models import Sequential
model = Sequential()
model.add(LSTM(10, input_shape=(6, 5)) )
model.add(Dense(512, activation='relu'))
model.add(Dense(6, activation='relu'))

model.compile(optimizer='adam', loss='mse', metrics=['mse'])

model.fit(x_data, y_data, batch_size=1, epochs=100, verbose=1)
y_pred = model.predict(x_data)
print(y_pred)
y_pred = np.argmax(y_pred, axis=2)
print(y_pred)
# result_str = [idxx2char[c] for c in np.squeeze(result)]
# print("\nPrediction str: ", ''.join(result_str))