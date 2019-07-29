import numpy as np

#1. 훈련 데이터 

x_train = np.arange(1, 101)
y_train = np.arange(501, 601)



x_test = np.arange(1001, 1100)
y_test = np.arange(1101, 1200)

 
#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()


model.add(Dense(5, input_dim = 1, activation ='relu'))
model.add(Dense(12))
model.add(Dense(3))
#model.add(Dense(14))
model.add(Dense(3))
#model.add(Dense(4))
model.add(Dense(1))


model.summary()
#model.summary() # param은 dense 가 5와 3일떄는 input weight 5, bias 1, x 3

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
#model.fit(x, y, epochs=100, batch_size=11)
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=3)
print('acc:', acc)

y_predict = model.predict(x_test)
print(y_predict)