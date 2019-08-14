from keras.datasets import cifar10
from keras.models import Model, Input
from keras.layers import Dense

(train, __,), (test, __,) = cifar10.load_data()
train = train.astype('float32') / 255
test = test.astype('float32') / 255

train = train[:10]
test = test[:10]
print(train.shape)
print(test.shape)


def build_network(optimizer='adam'):
    inputs = Input(shape=(32, 32, 3))

#     x = Dense(512, activation='relu')(inputs)
    encoded = Dense(32, activation='relu')(inputs)

    x = Dense(32, activation='relu')(encoded)
#     x = Dense(512, activation='relu')(x)   
    decoded = Dense(3, activation='sigmoid')(x)



    model = Model(inputs = inputs, outputs=decoded)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model

# from keras.wrappers.scikit_learn import KerasClassifier
# model = KerasClassifier(build_fn=build_network, verbose=1)
from keras.callbacks import EarlyStopping
es_cb = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')
model = build_network()

history = model.fit(train, train, batch_size=1024, epochs=50, shuffle=True, validation_data=(test, test), verbose=2, callbacks=[es_cb])

loss, acc = model.evaluate(test, test)
print(loss, acc)

pred = model.predict(test)

def showOrigDec(orig, dec, num=10):
    import matplotlib.pyplot as plt
    n = num
    plt.figure(figsize=(20, 4))

    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i+1)
        plt.imshow(orig[i].reshape(32, 32, 3))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i +1 + n)
        plt.imshow(dec[i].reshape(32, 32, 3))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


showOrigDec(test, pred)

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