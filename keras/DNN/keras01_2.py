import numpy as np

#1. 훈련 데이터 
x = np.array([1,2,3])
y = np.array([1,2,3])

x2 = np.array([7,8,9])

 
#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(4, input_dim = 1, activation ='relu'))
model.add(Dense(4))
model.add(Dense(12))
model.add(Dense(8))
model.add(Dense(2))
model.add(Dense(1))

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(x, y, epochs=500, batch_size=1)

#4. 평가 예측
loss, acc = model.evaluate(x, y, batch_size=1)
print('acc:', acc)

y_predict = model.predict(x2)
print(y_predict)