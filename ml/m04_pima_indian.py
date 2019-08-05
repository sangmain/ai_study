from keras.models import Sequential
from keras.layers import Dense
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

#모델의 생성
model = Sequential()
model.add(Dense(128, input_dim=8, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#모델 컴파일
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#모델 실행
model.fit(x, y, epochs=200, batch_size=10)

#결과 출력
print("\n Accuracy: %.4f" % (model.evaluate(x,y)[1]))
