import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Softmax
from sklearn.metrics import accuracy_score

#1. 데이터
learn_data = np.array([[0,0], [1,0], [0,1], [1,1]])
learn_label = np.array([0,0,0,1])

#2. 모델
clf = Sequential()   #clf = model

#3. 학습
# clf.add()
clf.add(Dense(512, input_shape = (2,), activation='relu'))
clf.add(Dense(1, activation='sigmoid'))


clf.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
clf.fit(learn_data, learn_label, epochs=100, batch_size=1)

#4. 평가
x_test = np.array([[0,0], [1,0], [0,1], [1,1]])
y_pred = clf.predict(x_test)

print(x_test, "예측 결과", y_pred)
acc = clf.evaluate(learn_data, learn_label, batch_size=1)
print('acc = ', acc[1])