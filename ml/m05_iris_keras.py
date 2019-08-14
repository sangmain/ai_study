

import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
import numpy as np

#붓꽃데이터 읽어들이기
colnames = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Name'] 
iris_data = pd.read_csv("./data/iris.csv", names= colnames, encoding='utf-8')

#붓꽃 데이터를 레이블과 입력 데이터로 뷴리하기
y = iris_data.loc[:, "Name"]
x = iris_data.loc[:, ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]]

# string one hot encoding
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)

y = to_categorical(y).astype(int)

#학습 전용과 테스트 전용 분리하기
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, train_size = 0.8, shuffle=True)

#학습하기
model = Sequential()

model.add(Dense(1000, input_dim = 4))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'] )
model.fit(x_train, y_train, epochs=50, batch_size=1)
#평가하기
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
print(y_pred)
y_pred = encoder.inverse_transform(y_pred)
print(y_pred)
acc = model.evaluate(x_test, y_test, batch_size=1)
print("정답률: ",acc[1] )
