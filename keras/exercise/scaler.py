import numpy as np

from keras.models import Sequential
from keras.layers import Dense

data = np.array([1,2,3,4,5,6,7,8,1100000, 16,17,18,19,20,21,22,23, 400000])

x = data
y = data

print(x.shape)
x = np.transpose(x)
y = np.transpose(y)
print(y.shape)


#이건 적어도 2d ARRay 에다가 쓰는것 같다
from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)
print(x.shape)


model = Sequential()

model.add(Dense(10000, input_shape=(1,), activation='relu'))
model.add(Dense(10))
model.add(Dense(1))


model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
model.fit(x, y, epochs=100, batch_size=1, verbose=2)

y_pred = model.predict(x)

print(y_pred)