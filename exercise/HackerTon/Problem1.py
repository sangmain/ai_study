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
from keras.models import Model, Input
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten

#신경망 정의
def build_network(keep_prob=0.5, optimizer='adam', node1=32, node2=60):
    inputs = Input(shape=(32,32,3), name='input')

    x1 = Conv2D(node1, kernel_size=(3,3), padding='same', activation='relu', name='hidden1')(inputs)
    # max1 = MaxPooling2D(pool_size=(2,2))(x1)
    dp1 = Dropout(keep_prob)(x1)

    f1 = Flatten()(dp1)
    x2 = Dense(node2, activation='relu')(f1)
    # dp2 = Dropout(keep_prob)(x2)
    prediction = Dense(10, activation='softmax')(x2)

    model = Model(inputs=inputs, outputs=prediction)
    model.compile(loss= 'categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

#하이퍼파람
def create_hyperparameter():
    batches =[32,64,128,256,500]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = np.linspace(0.1, 0.5, 5)
    node1 = [10,16,32,64]
    node2 = [5,10,30,100,200]
    return{"batch_size": batches, "optimizer": optimizers, "keep_prob":dropout, "node1":node1, "node2":node2}



# 모델 생성
from keras.wrappers.scikit_learn import KerasClassifier
model = KerasClassifier(build_fn=build_network, verbose=1)

#파람 생성
hyperparameters = create_hyperparameter()

#최적값 찾기
from sklearn.model_selection import RandomizedSearchCV
rs = RandomizedSearchCV(estimator=model, param_distributions=hyperparameters, cv=3, verbose=1)

rs.fit(x_ext, y_ext)
print("최적: ", rs.best_params_)
print("train: ",rs.score(x_ext, y_ext))
print("test: ",rs.score(x_test, y_test))