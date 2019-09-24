import numpy as np

#1. 훈련 데이터 
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])

x_test = np.array([11,12,13,14,15,16,17,18,19,20])
y_test = np.array([11,12,13,14,15,16,17,18,19,20])

x3= np.array([101, 102, 103, 104, 105, 106])
x4= np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
x5= np.array(range(30, 50))
#열이 우선이다 행은 무시된다

#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()


model.add(Dense(5, input_shape = (1, ), activation ='relu'))
#model.add(Dense(5, input_dim = 1, activation ='relu'))
model.add(Dense(12))
model.add(Dense(3))
#model.add(Dense(14))
model.add(Dense(3))
#model.add(Dense(4))
model.add(Dense(1))

model.summary()
# #model.summary() # param은 dense 가 5와 3일떄는 input weight 5, bias 1, x 3

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
#model.fit(x, y, epochs=100, batch_size=11)
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=3)
print('acc:', acc)


print(x5)
y_predict = model.predict(x5)
print(y_predict)