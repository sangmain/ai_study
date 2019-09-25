from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization

#기온 데이터 읽어들이기
df = pd.read_csv('./data/tem10y.csv', encoding='utf-8')

#데이터를 학습 전용과 테스트 전용으로 분리하기
train_year = (df['연'] <= 2015)
test_year = (df["연"] >= 2016)
interval = 6

# print(train_year)
# print(test_year)
#과거 6명의 데이터를 기반으로 학습할 데이터 만들기
def make_data(data):
    x = []
    y = []

    temps = list(data["기온"])
    for i in range(len(temps)):
        if i < interval: continue
        y.append(temps[i])
        xa = []

        for p in range(interval):
            d = i + p - interval
            xa.append(temps[d])
        x.append(xa)
    return (x, y)

x_train, y_train =  make_data(df[train_year])
x_test, y_test = make_data(df[test_year])

from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = StandardScaler()
# scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.utils import shuffle
x_train, y_train = shuffle(x_train, y_train, random_state=5)
# # x_test, y_test = shuffle(x_test, y_test, random_state=5)

x_train = np.array(x_train)
x_train = x_train.reshape(3646, 6, 1)
x_test = np.array(x_test)
x_test = x_test.reshape(360, 6, 1)
print(x_train.shape)

#직선 회귀 분석하기



model = Sequential()

model.add(LSTM(25, input_shape=(6,1), activation='softsign'))#, return_sequences=True))
# model.add(LSTM(50))
# model.add(Dropout(0.5))

model.add(Dense(20, activation='relu'))
model.add(BatchNormalization())
# model.add(Dropout(0.5))
model.add(Dense(30, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(60, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(10, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(1))

model.summary()

from keras.callbacks import EarlyStopping
early = EarlyStopping(monitor='val_loss', patience=30, mode='auto')
model.compile(loss='mse', optimizer='rmsprop', metrics=['mse'])
model.fit(x_train, y_train, epochs=300, batch_size=16, validation_data=(x_test, y_test), callbacks=[early])

y_pred = model.predict(x_test)

from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE: ", RMSE(y_test, y_pred))

#R2 구하기
from sklearn.metrics import r2_score

r2_y_predict = r2_score(y_test, y_pred)
print("R2: ", r2_y_predict)


# #결과를 그래프로 그리기
# # plt.figure(figsize=(10, 6), dpi=100)
# # plt.plot(y_test, c='r')
# # plt.plot(y_pred, c='b')
# # plt.savefig('tenki-kion-lr.png')
# # plt.show()