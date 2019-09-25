import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from keras.models import Model, Input
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
#데이터 읽어들이기
wine = pd.read_csv("./data/winequality-white.csv", sep=';', encoding='utf-8')

#데이터를 레이블과 데이터로 분리하기
y = wine["quality"]
# x = wine.iloc[:, 0:4]
x = wine.drop(["quality"], axis=1)
y = to_categorical(y)
print(np.argmax(y))
# print(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=66)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train)
#학습하기
def build_network():
    inputs = Input(shape=(11,), name='input')

    x = Dense(512, activation='relu', name='hidden1')(inputs)
    x1 = Dense(128, name='hidden2')(x)
    x2 = Dense(64, name='hidden3')(x1)
    x3 = Dense(32, name='hidden4')(x2)
    x4 = Dense(13, name='hidden5')(x3)
    prediction = Dense(10, activation='softmax')(x4)

    model = Model(inputs = inputs, outputs=prediction)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

from keras.wrappers.scikit_learn import KerasClassifier
model = KerasClassifier(build_fn=build_network, verbose=1, epochs=100, batch_size = 64)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
pipe.fit(x_train, y_train)

print(pipe.score(x_test, y_test))
print(model.score(x_test, y_test))