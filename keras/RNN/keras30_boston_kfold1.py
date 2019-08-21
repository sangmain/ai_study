from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()


print(train_data.shape)
print(train_targets.shape)
print(test_data.shape)

mean = train_data.mean(axis = 0)
train_data -= mean
std = train_data.std(axis=0)
# train_data /= std

test_data -= mean
test_data /= std

from keras import models
from keras import layers

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape = (train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))

    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


import numpy as np
k = 5
num_val_samples = len(train_data) // k
num_epochs = 1
all_scores = []

for i in range(k):
    print('처리중인 폴드', i)
    val_data = train_data[i * num_val_samples : (i+1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples : (i+1) * num_val_samples]

    partial_train_data = np.concatenate(
        [train_data[ : i * num_val_samples],
         train_data[(i+1) * num_val_samples:]],
          axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[ : i * num_val_samples],
         train_targets[(i+1) * num_val_samples:]],
          axis=0)

    print(partial_train_data.shape)
    model = build_model()

    model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=1, verbose=5)

    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose = 0)
    all_scores.append(val_mae)

print(all_scores)
print(np.mean(all_scores))