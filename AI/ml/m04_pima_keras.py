from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, BatchNormalization
import numpy
import tensorflow as tf

#시드값 생성
seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

#데이터 로드
dataset = numpy.loadtxt('./data/pima-indians-diabetes.csv', delimiter=',')
x = dataset[:, 0:8]
y = dataset[:, 8]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, test_size=0.2)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = StandardScaler()
# scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#모델의 생성
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#모델 컴파일
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

early = EarlyStopping(monitor='val_acc', patience=30, mode='auto')
#모델 실행
model.fit(x_train, y_train, epochs=20, batch_size=8, callbacks=[early], validation_data=(x_test, y_test))

#결과 출력
print("\n Accuracy: %.4f" % (model.evaluate(x_test,y_test)[1]))
