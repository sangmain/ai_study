import numpy as np

#1. 훈련 데이터 
x = np.array(range(1,101))
y = np.array(range(1,101))

#print(x)
x_train, x_val, x_test = np.split(x, [60,80])
y_train, y_val, y_test = np.split(y, [60,80])

#열이 우선이다 행은 무시된다

#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()


model.add(Dense(5, input_shape = (1, ), activation ='relu'))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(1))

#model.summary()
# #model.summary() # param은 dense 가 5와 3일떄는 input weight 5, bias 1, x 3

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
#model.fit(x, y, epochs=100, batch_size=11)
model.fit(x_train, y_train, epochs=100, batch_size=1)
 #       validation_data= (x_val,  y_val))

#4. 평가 예측
# loss, acc = model.evaluate(x_test, y_test, batch_size=3)
# print('acc:', acc)

y_predict = model.predict(x_test)
print(y_predict)

#RMSE 구하기
from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE: ", RMSE(y_test, y_predict))

#R2 구하기
from sklearn.metrics import r2_score

r2_y_predict = r2_score(y_test, y_predict)
print("R2: ", r2_y_predict)
