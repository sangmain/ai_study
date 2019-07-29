import numpy as np


data = np.array([1,2,3,4,5,6,7,8,9,10])

x = data[:7]
y = data[:7]

x_test =  data[7:]
y_test =  data[7:]


from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(10000, input_dim=1, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

from keras.callbacks import EarlyStopping
early = EarlyStopping(monitor='mse', patience=100, mode='auto')

hist = model.fit(x, y, epochs=100, batch_size=1, verbose=2, callbacks=[early])


loss, acc = model.evaluate(x, y, batch_size=1)
print('loss: ', loss)
print('acc: ', acc)
y_pred = model.predict(x_test)

print(y_pred)

model.summary()

from sklearn.metrics import r2_score

r2_pred = r2_score(y_test, y_pred)
print("R2: ", r2_pred)
