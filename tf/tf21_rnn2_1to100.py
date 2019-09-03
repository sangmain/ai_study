import numpy as np
x = np.arange(100)
print(x)

def split_5(seq, size):
    aaa = []
    for i in range(len(seq)-size + 1):
        subset = seq[i: (i + size)] 
        aaa.append(subset)

    print(type(aaa))
    return np.array(aaa)

dataset = split_5(x, 10)

np.random.shuffle(dataset)
print(dataset)
x_train = dataset[:, :-3]
y_train = dataset[:, -3:]

print(x_train[0])
print(y_train[0])

print(x_train.shape)
x_train = x_train.reshape(-1, 6, 3)

# from keras.models import Sequential
# from keras.layers import LSTM, Dense, BatchNormalization

# model = Sequential()
# model.add(LSTM(10, input_shape=(6,3)))
# model.add(Dense(512, activation='relu'))
# # model.add(BatchNormalization()
# model.add(Dense(1))

# model.compile(optimizer='adam', loss='mse', metrics=['accuracy'] )
# model.fit(x_train, y_train, batch_size=5, epochs=200)

# # x_test =  [[101,102,103,104,105,106]]
# x_test =  np.array([[90,91,92,93,94,95]])
# y_test = [96, 97, 98]
# print(x_test.shape)
# x_test = x_test.reshape(-1, 6, 1)
# y_pred = model.predict(x_test, batch_size=1)
# print(y_pred)

# from sklearn.metrics import mean_squared_error

# def RMSE(y_test, y_predict):
#     return np.sqrt(mean_squared_error(y_test, y_pred))

# print("RMSE: ", RMSE(y_test, y_pred))