import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

a = np.array(range(1, 101))

size = 8

def split_5(seq, size):
    aaa = []
    for i in range(len(a)-size+1):
        subset = a[i:(i+size)]
        aaa.append(subset)

    print(type(aaa))
    return np.array(aaa)

dataset = split_5(a, size)
print("================")
# print(dataset)
x = dataset[:, 0:4]
y = dataset[:, 4:8]


# x_train = np.reshape(x_train, (6,4,1))
# x = np.reshape(x, (len(a)-size+1,4,1))
x = x.reshape(-1, 2, 2)
print(x.shape)



from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split( x, y, random_state = 66, test_size = 0.8)
# x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, random_state = 66, test_size = 0.5)

print(x_test.shape)
print(y_test.shape)

#모델 구성
model = Sequential()

model.add(LSTM(32, input_shape=(2,2), return_sequences=False))
# model.add(LSTM(10, return_sequences=True))
# model.add(LSTM(10, return_sequences=True))
# model.add(LSTM(10, return_sequences=True))
# model.add(LSTM(10, return_sequences=True))
# model.add(LSTM(10, return_sequences=True))
# model.add(LSTM(10))

# model.add(Dense(5, activation='relu'))
# model.add(Dense(5, activation='relu'))
# model.add(Dense(5, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4))

model.summary()

#훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])

from keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='loss', patience=30, mode='auto')
model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1, callbacks=[early_stop])

loss, acc = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)

print('loss: ', loss)
print('acc: ', acc)
print('y_predict(x_test): \n', y_predict)


from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE: ", RMSE(y_test, y_predict))

#R2 구하기
from sklearn.metrics import r2_score

r2_y_predict = r2_score(y_test, y_predict)
print("R2: ", r2_y_predict)


