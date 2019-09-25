from keras.datasets import mnist
import numpy as np

(train,_), (test,_) = mnist.load_data()

train = train[:]
test = test[:]

train = train.astype('float32') / 255
test = test.astype('float32') / 255

print(train.shape)
print(test.shape)

train = train.reshape( len(train), np.prod(train.shape[1:]) )
test = test.reshape( len(test), np.prod(test.shape[1:]) )

print(train.shape)
print(test.shape)

from keras.layers import Input, Dense,Dropout, BatchNormalization
from keras.models import Model

encoding_dim = 32#

# 함수형 모델.

    # 플레이스 폴더
input_img = Input(shape=(784,))
# 입력의 인코딩된 표현
e1 = Dense(400, activation='relu')(input_img)
dp1 = BatchNormalization()(e1)
e2 = Dense(encoding_dim, activation='relu')(dp1)
dp2 =BatchNormalization()(e2)
e3 = Dense(64, activation='relu')(dp2)
dp3 =BatchNormalization()(e3)

e4 = Dense(600, activation='relu')(dp3)
dp4 = BatchNormalization()(e4)

d0 = Dense(64, activation='relu')(dp4)
dp5 =BatchNormalization()(d0)
d1 = Dense(encoding_dim, activation='relu')(dp5)
dp6 = BatchNormalization()(d1)
d2 = Dense(400, activation='relu')(dp6)

# e1 = Dense(64, activation='relu', kernel_regularizer= regularizers.l1(0.0001))(input_img)
# # dp1 = BatchNormalization()
# e2 = Dense(encoding_dim, activation='relu')(e1)
# # dp2 = Dropout(0.5)(e2)
# e3 = Dense(600, activation='relu')(e2)
# # dp3 = Dropout(0.5)(e3)
# d1 = Dense(encoding_dim, activation='relu')(e3)
# # dp4 = Dropout(0.5)(d1)
# d2 = Dense(64, activation='relu')(d1)


    # 손실있는 재 구성
decoded = Dense(784, activation='sigmoid')(d2)
# decoded = Dense(784, activation='relu')(encoded)
    # 입력을 입력의 재구성으로 매핑할 모델
auto_encoder = Model(input_img, decoded) # 784 > 32 > 784

    # 이 모델은 입력을 입력의 인코딩된 입력의 표현으로 매핑
encoder = Model(input_img, e1) #  784 > 32

encoded_input = Input(shape=(400,))

decoder_layer = auto_encoder.layers[-1]

decoder = Model(encoded_input, decoder_layer(encoded_input)) # 32 > 784

auto_encoder.summary()
encoder.summary()
decoder.summary()

auto_encoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
history = auto_encoder.fit(train, train, epochs= 50, batch_size=256, shuffle=True, validation_data=(test, test))
# 입력(train)으로 결과 값(train)을 학습한다.

encoded_imgs = encoder.predict(test)
decoded_imgs = decoder.predict(encoded_imgs)

print(encoded_imgs)
print(decoded_imgs)
print(encoded_imgs.shape)
print(decoded_imgs.shape)

# 시각화
import matplotlib.pyplot as plt

n = 10
plt.figure(figsize=(20,4))

for i in range(n):

    # 윈본 데이터
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # 재구성된 데이터
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()

def plot_acc(history, title = None):
    if not isinstance(history, dict):
        history =  history.history
    
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])

    if title is not None:
        plt.title(title)
    
    plt.ylabel('Accracy')
    plt.xlabel('Epoch')
    plt.legend(['Traing data'], 'Validation data', loc = 0)

def plot_loss(history, title = None):
    if not isinstance(history, dict):
        history =  history.history
    
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])

    if title is not None:
        plt.title(title)
    
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Traing data'], 'Validation data', loc = 0)

plot_acc(history, '(a) 학습 경과에 따른 정확도 변화 추이')
plt.show()
plot_loss(history, '(b) 학습 경과에 따른 손실값 변화 추이')
plt.show()


loss, acc = auto_encoder.evaluate(test, test)
print(loss, acc)