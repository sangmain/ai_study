import numpy as np
import pandas as pd

data = pd.read_csv("./data/test0822.csv", delimiter=",")


row_per_col = 50

x = data.iloc[:3113, 1:9]
x = np.array(x)
print(x.shape)

test = data.iloc[3113-row_per_col:3113, 1:9]
print(test.shape)



x_len = 50
y_len = 5
sequence_length = x_len + y_len
size = row_per_col + 5

result = []
for i in range(len(x) - sequence_length + 1):
    idk = []
    idk = x[i:i+size]
    result.append(idk)

result = np.array(result)

x = result[:, :50, :]
y = result[:, 50:, :]

print(x.shape)
print(y.shape)

x = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = StandardScaler()
# scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)
test = scaler.transform(test)

x = x.reshape(3059, 50, 8)
test = test.reshape(1, -1, 8)

y = y.reshape(y.shape[0], y.shape[1] * y.shape[2])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split( x, y, random_state = 66, test_size = 0.2)

print(x_train.shape)
print(y_train.shape)

from keras.models import Sequential
from keras.layers import LSTM, Dense

model = Sequential()

model.add(LSTM(50, input_shape = (50,8), activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(512, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
model.add(Dense(40, activation='relu'))

model.compile(optimizer='adam', loss='mse', metrics=['mse'])
model.fit(x_train, y_train, epochs=300, batch_size=128, verbose=2)


pred = model.predict(x_test, batch_size=64)
print(pred)

from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, pred)


print("x_test: ", x_test[0])
print("y_test: ",y_test[0])
print("pred: ", pred[0])

print("R2: ", r2_y_predict)
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE: ", RMSE(y_test, pred))



############### 평가
print("\n\n\n\n\n")
pred = model.predict(test, batch_size=2)
print(pred)

pred = pred.reshape(5, -1)
pred = np.round(pred)
print(pred)
dataframe = pd.DataFrame(pred)
dataframe.to_csv('./test0822/test0822.csv', header = False, index=False)