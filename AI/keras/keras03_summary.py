import numpy as np

#1. 훈련 데이터 
x = np.array([1,2,3,4,5,6,7,8,9])
y = np.array([1,2,3,4,5,6,7,8,9])

# x2 = np.array([4,5,6])

 
#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(5, input_dim = 1, activation ='relu'))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(1))


model.summary() # param은 dense 가 5와 3일떄는 input weight 5, bias 1, x 3

# #3. 훈련
# model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
# #model.fit(x, y, epochs=100, batch_size=11)
# model.fit(x, y, epochs=100)

# #4. 평가 예측
# loss, acc = model.evaluate(x, y, batch_size=1)
# print('acc:', acc)

# y_predict = model.predict(x2)
# print(y_predict)