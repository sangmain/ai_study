from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping

import numpy
import os
import tensorflow as tf

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

digit = X_train[2000]

# X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255
# X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255
X_train = X_train.reshape(X_train.shape[0], 28*28).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 28*28).astype('float32') / 255

print(Y_train.shape)
print(Y_test.shape)

Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)

# print(X_train.shape)
# print(X_test.shape)

print(Y_train.shape)
print(Y_test.shape)



from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout, Input
import numpy as np

def build_network(keep_prob=0.5, optimizer='adam'):
    inputs = Input(shape=(784, ), name='input')

    x = Dense(512, activation ='relu', name = 'hidden1')(inputs)
    x = Dropout(keep_prob)(x)
    x1 = Dense(256, activation ='relu', name = 'hidden2')(x)
    x1 = Dropout(keep_prob)(x1)
    x2 = Dense(128, activation ='relu', name = 'hidden3')(x1)
    x2 = Dropout(keep_prob)(x2)
    prediction = Dense(10, activation='softmax', name='output')(x2)
    model = Model(inputs=inputs, outputs=prediction)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def create_hyperparameter():
    batches =[10,20,30,40,50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = np.linspace(0.1, 0.5, 5)
    return{"batch_size": batches, "optimizer": optimizers, "keep_prob":dropout}


from keras.wrappers.scikit_learn import KerasClassifier

model = KerasClassifier(build_fn=build_network, verbose=1)

hyperparameters = create_hyperparameter()

from sklearn.model_selection import GridSearchCV

search = GridSearchCV(estimator=model, param_grid= hyperparameters, cv=3, verbose=1)

search.fit(X_train, Y_train)

print(search.best_params_)