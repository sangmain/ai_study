from keras.models import Model, Input
from keras.layers import Dense, Dropout, BatchNormalization
from keras.utils import to_categorical
from sklearn.datasets import load_breast_cancer


import numpy as np


cancer = load_breast_cancer()

x = cancer.data
y = cancer.target   

# y = to_categorical(y)
# print(y)
# print(x.shape)
# print(y.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=66)

print(x_train.shape)
print(y_train.shape)

def build_network(keep_prob = 0.5, optimizer='adam'):
    inputs = Input(shape=(30,), name='input')

    # x = Dense(512, activation='relu', name= 'hidden1')(inputs)
    x = Dense(512, activation='relu', name= 'hidden1')(inputs)
    prediction = Dense(1, activation='softmax', name='output')(x)

    model = Model(inputs=inputs, outputs=prediction)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model

def create_hyperparameter():
    batches = [10, 50, 100]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = np.linspace(0.1, 0.25, 5)
    return{"batch_size": batches, "optimizer": optimizers, "keep_prob":dropout}




from keras.wrappers.scikit_learn import KerasClassifier
model = KerasClassifier(build_fn=build_network, verbose=1)

hyperparameters = create_hyperparameter()

from sklearn.model_selection import RandomizedSearchCV
search = RandomizedSearchCV(model, hyperparameters)

search.fit(x_train, y_train)

print(search.best_params_)