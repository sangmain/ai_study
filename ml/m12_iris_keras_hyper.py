import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model, Input
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

y = to_categorical(y)

#학습 전용과 테스트 전용 분리하기
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, train_size = 0.8, shuffle=True)

#학습하기

def build_network(optimizer='adam'):
    inputs = Input(shape=(4,), name='input')

    x = Dense(1000, activation='relu', name = 'hidden1')(inputs)
    prediction = Dense(3, activation='softmax', name = 'output')(x)

    model = Model(inputs=inputs, outputs=prediction)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def create_hyperparameter():
    batches = [10, 20, 30, 40, 50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    return{"batch_size": batches, "optimizer": optimizers}



from keras.wrappers.scikit_learn import KerasClassifier
model = KerasClassifier(build_fn=build_network, verbose=1)

hyperparameters = create_hyperparameter()

from sklearn.model_selection import RandomizedSearchCV
search = RandomizedSearchCV(estimator=model, param_distributions=hyperparameters)

search.fit(x_train, y_train)

#평가하기
print(search.best_params_)
