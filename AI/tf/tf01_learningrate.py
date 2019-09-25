import numpy as np

#1. 훈련 데이터 
x = np.array([1,2,3,4])
y = np.array([1,2,3,4])

x2 = np.array([7,8,9])

 
#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(5, input_dim = 1, activation ='relu'))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(1))

#3. 훈련
from keras.optimizers import * #import SGD, Adagrad, Adadelt, Adam, RMSprop

optimizer = SGD(lr=0.01, clipnorm=1.) 
# optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
# optimizer = Adagrad(lr=0.01, epsilon=None, decay=0.0)
# optimizer = Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
# optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
# optimizer = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
# optimizer = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)


# model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.compile(loss='mse', optimizer= optimizer, metrics=['mse'])

model.fit(x, y, epochs=100, batch_size=1, verbose=0)

#4. 평가 예측
mse, _ = model.evaluate(x, y, batch_size=1)
print('mse: ', mse)

pred1 = model.predict([1.5,2.5,3.5])
print(pred1)