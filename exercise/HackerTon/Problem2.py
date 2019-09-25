from keras.datasets import cifar10
from keras.utils import np_utils
import matplotlib.pyplot as plt

import numpy as np

#데이터 불러오기
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#300개로 나누기
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.006, random_state=66)
x_test = x_test[:300]
y_test = y_test[:300]


#이미지 제너레이터
def generate_data(x_train, y_train):
    from keras.preprocessing.image import ImageDataGenerator
    from keras.preprocessing import image

    datagen = ImageDataGenerator(rotation_range=20,
                                    width_shift_range = 0.1,
                                    height_shift_range=0.1,
                                    shear_range=0.1,
                                    zoom_range=0.1,
                                    horizontal_flip=True,
                                    fill_mode='nearest'
                                )

    x_ext = []
    y_ext = []
    for i in range(x_train.shape[0]):
        img = x_train[i]
        img = img.reshape((1,) + img.shape)


        j = 0
        for batch in datagen.flow(img, batch_size=1):
            x_ext.append(batch[0])
            y_ext.append(y_train[i])
            if j == 4:
                break
            j += 1

    x_ext = np.array(x_ext)
    y_ext = np.array(y_ext)
    return x_ext, y_ext

#데이터 갯수 늘리기
x_ext, y_ext = generate_data(x_train, y_train)

#변환
y_ext = np_utils.to_categorical(y_ext, 10)
y_test = np_utils.to_categorical(y_test, 10)

x_ext = x_ext.astype('float32')
x_test = x_test.astype('float32')
x_ext /= 255
x_test /= 255


#셔플
s = np.arange(x_ext.shape[0])
np.random.shuffle(s)
x_ext = x_ext[s]
y_ext = y_ext[s]

print("x: ", x_ext.shape)
print("y: ", y_ext.shape)

#모델
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten

model = Sequential()
model.add(Conv2D(32, (3,3), padding='same', input_shape=(32, 32, 3), activation='relu'))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(100,activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss= 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

#훈련
history = model.fit(x_ext, y_ext, batch_size=32, epochs=100, validation_data=(x_test, y_test),verbose = 0)

score = model.evaluate(x_test, y_test, batch_size = 32, verbose= 0)

print('Testing...')
print('\nTest Score', score[0])
print('Test Accuracy', score[1])

#그래프 출력
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()