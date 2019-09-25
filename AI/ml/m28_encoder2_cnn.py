from keras.datasets import mnist
import numpy as np

(train,_), (test,_) = mnist.load_data()

train = train[:]
test = test[:]

train = train.reshape(train.shape[0], 28, 28, 1).astype('float32') / 255
test = test.reshape(test.shape[0], 28, 28, 1).astype('float32') / 255

from keras.layers import Input, Dense,Dropout, BatchNormalization, Conv2D, MaxPooling2D, Flatten, UpSampling2D
from keras.models import Model

encoding_dim = 32#

# 함수형 모델.

    # 플레이스 폴더
input_img = Input(shape=(28,28,1))
# 입력의 인코딩된 표현
e1 = Conv2D(32, kernel_size=(3,3), activation='relu', padding='same')(input_img)
e2 = Conv2D(64, kernel_size=(3,3), activation='relu', padding='same')(e1)

e3 = Conv2D(32, kernel_size=(3,3), activation='relu', padding='same')(e2)

# e4 = Flatten()(e3)
decoded = Conv2D(1, kernel_size=(3,3),activation='sigmoid', padding='same')(e1)
# decoded = Dense(784, activation='relu')(encoded)
    # 입력을 입력의 재구성으로 매핑할 모델
auto_encoder = Model(input_img, decoded) # 784 > 32 > 784

auto_encoder.summary()
# encoder.summary()
# decoder.summary()

auto_encoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
history = auto_encoder.fit(train, train, epochs= 50, batch_size=256, shuffle=True, validation_data=(test, test))
loss, acc = auto_encoder.evaluate(test, test)
print(loss, acc)

# 입력(train)으로 결과 값(train)을 학습한다.

# encoded_imgs = encoder.predict(test)
# decoded_imgs = decoder.predict(encoded_imgs)

# print(encoded_imgs)
# print(decoded_imgs)
# print(encoded_imgs.shape)
# print(decoded_imgs.shape)

# # 시각화
# import matplotlib.pyplot as plt

# n = 10
# plt.figure(figsize=(20,4))

# for i in range(n):

#     # 윈본 데이터
#     ax = plt.subplot(2, n, i + 1)
#     plt.imshow(test[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)

#     # 재구성된 데이터
#     ax = plt.subplot(2, n, i + 1)
#     plt.imshow(decoded_imgs[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)

# plt.show()

# def plot_acc(history, title = None):
#     if not isinstance(history, dict):
#         history =  history.history
    
#     plt.plot(history['acc'])
#     plt.plot(history['val_acc'])

#     if title is not None:
#         plt.title(title)
    
#     plt.ylabel('Accracy')
#     plt.xlabel('Epoch')
#     plt.legend(['Traing data'], 'Validation data', loc = 0)

# def plot_loss(history, title = None):
#     if not isinstance(history, dict):
#         history =  history.history
    
#     plt.plot(history['loss'])
#     plt.plot(history['val_loss'])

#     if title is not None:
#         plt.title(title)
    
#     plt.ylabel('Loss')
#     plt.xlabel('Epoch')
#     plt.legend(['Traing data'], 'Validation data', loc = 0)

# plot_acc(history, '(a) 학습 경과에 따른 정확도 변화 추이')
# plt.show()
# plot_loss(history, '(b) 학습 경과에 따른 손실값 변화 추이')
# plt.show()

