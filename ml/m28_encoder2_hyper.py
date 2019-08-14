from keras.datasets import mnist
import numpy as np

(train,_), (test,_) = mnist.load_data()

train = train[:1200]
test = test[:1200]

train = train.astype('float32') / 255
test = test.astype('float32') / 255

print(train.shape)
print(test.shape)

train = train.reshape( len(train), np.prod(train.shape[1:]) )
test = test.reshape( len(test), np.prod(test.shape[1:]) )

print(train.shape)
print(test.shape)

from keras.layers import Input, Dense, Dropout
from keras.models import Model

def build_network(keep_prob=0.5, optimizer='adam'):
    encoding_dim = 32#

    # 함수형 모델.
        # 플레이스 폴더
    input_img = Input(shape=(784,))
        # 입력의 인코딩된 표현
    encoded = Dense(encoding_dim, activation='relu')(input_img)
    dp1 = Dropout(keep_prob)(encoded)
    encoded1 = Dense(10, activation='relu')(dp1)

    dp2 = Dropout(keep_prob)(encoded1)
    encoded3 = Dense(encoding_dim, activation='relu')(dp2)
        # 손실있는 재 구성
    decoded = Dense(784, activation='sigmoid')(encoded3)

        # 입력을 입력의 재구성으로 매핑할 모델
    auto_encoder = Model(input_img, decoded) # 784 > 32 > 784

        # 이 모델은 입력을 입력의 인코딩된 입력의 표현으로 매핑
    # encoder = Model(input_img, encoded) #  784 > 32

    # encoded_input = Input(shape=(encoding_dim,))

    # decoder_layer = auto_encoder.layers[-1]

    # decoder = Model(encoded_input, decoder_layer(encoded_input)) # 32 > 784
    
    auto_encoder.summary()
    # encoder.summary()
    # decoder.summary()
    auto_encoder.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return auto_encoder

# history = auto_encoder.fit(train, train, epochs= 50, batch_size=256, shuffle=True, validation_data=(test, test))
# 입력(train)으로 결과 값(train)을 학습한다.

def create_hyperparameter():
    batches = [64, 128, 256, 500]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = np.linspace(0.1, 0.5, 5)
    # activation = ['relu', 'sigmoid']
    return {"batch_size":batches, "optimizer":optimizers, "keep_prob":dropout}


from sklearn.model_selection import KFold, RandomizedSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
model = KerasClassifier(build_fn=build_network, verbose=2)
parameters = create_hyperparameter()
clf = RandomizedSearchCV(estimator= model, param_distributions=parameters)

clf.fit(train, train)
print(clf.best_params_)
print(clf.score(test, test))

pred = clf.predict(test)
print(pred.shape)


# 시각화
import matplotlib.pyplot as plt

n = 10
plt.figure(figsize=(20,4))

for i in range(n):

    # 윈본 데이터
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(test[i].reshape(28, 28))
    plt.gray
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # 재구성된 데이터
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray
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