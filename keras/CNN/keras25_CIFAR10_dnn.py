from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense

from keras import models

BATCH_SIZE = 128
NB_EPOCH = 20
NB_CLASSES = 10
VERBOSE = 1
VALIDATION_SPLIT = 0.2
size = 32
channel = 3


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

model = Sequential()

dimension = size*size*channel

model.add(Dense(512, activation='relu', input_shape=(dimension,)))

model.add(Dense(10, activation='softmax'))

model.compile(optimizer = 'rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

x_train = x_train.reshape((x_train.shape[0], dimension))
x_test = x_test.reshape((x_test.shape[0], dimension))

mean = np.mean(x_train,axis=(0,1,2,3))
std = np.std(x_train,axis=(0,1,2,3))
x_train = (x_train-mean)/(std+1e-7)
x_test = (x_test-mean)/(std+1e-7)

y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH,
                     validation_split=VALIDATION_SPLIT, verbose = VERBOSE)


print('Testing...')
score = model.evaluate(x_test, y_test, batch_size = BATCH_SIZE, verbose= VERBOSE)
print('\nTest Score', score[0])
print('Test Accuracy', score[1])

