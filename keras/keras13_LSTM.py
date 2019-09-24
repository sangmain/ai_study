from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1. 데이터
x = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9],[8,9,10],[9,10,11],[10,11,12]])
y = array([4,5,6,7,8,9,10,11,12,13])

print("x.shape: ", x.shape)
print("y.shape: ", y.shape)


x = x.reshape((x.shape[0], x.shape[1], 1))
print("x.shape: ", x.shape)

#2. 모델구성
model = Sequential()

# model.add(LSTM(2, activation='relu', input_shape=(3,1)))
# model.add(Dense(6))

# model.add(Dense(2))
# model.add(Dense(5))


# model.add(Dense(1))


model.add(LSTM(100, activation='relu', input_shape=(3,1)))
model.add(Dense(100))
model.add(Dense(100))

model.add(Dense(300))

model.add(Dense(100))


model.add(Dense(1))

#model.summary()
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
model.fit(x, y, epochs=300, batch_size=1)

x_input = array([19,20,21]) # 1행 3열
x_input = x_input.reshape(1,3,1)


yhat = model.predict(x_input)
print(yhat)

# from sklearn.metrics import r2_score

# r2_y1_predict = r2_score(y_test, yhat)
# print("R2: ", r2_y1_predict)

