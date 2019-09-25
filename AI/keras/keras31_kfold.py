from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

# print(train_data.shape)
# print(train_targets.shape)
# print(test_data.shape)

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


# from sklearn.model_selection import KFold
# stf = KFold(n_splits=5)

# num_epochs = 1
# all_scores = []

# import numpy as np
# for train_index, target_index in stf.split(train_data, train_targets):
#     partial_train_data, val_data = train_data[train_index], train_data[target_index]
#     partial_train_targets, val_targets = train_targets[train_index], train_targets[target_index]
#     model = build_model()

#     model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=1, verbose=5)

#     val_mse, val_mae = model.evaluate(val_data, val_targets, verbose = 0)
#     all_scores.append(val_mae)

# print(all_scores)
# print(np.mean(all_scores))

# from sklearn.model_selection import StratifiedKFold
seed = 77

from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.model_selection import KFold, cross_val_score

model = KerasRegressor(build_fn = build_model, epochs=10, batch_size=1, verbose=1)

kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
results = cross_val_score(model, train_data, train_targets, cv=kfold)

import numpy as np
print(results)
print(np.mean(results))


# import numpy as np
# k = 5
# num_val_samples = len(train_data) // k
# num_epochs = 1
# all_scores = []

# for i in range(k):
#     print('처리중인 폴드', i)
#     val_data = train_data[i * num_val_samples : (i+1) * num_val_samples]
#     val_targets = train_targets[i * num_val_samples : (i+1) * num_val_samples]

#     partial_train_data = np.concatenate(
#         [train_data[ : i * num_val_samples],
#          train_data[(i+1) * num_val_samples:]],
#           axis=0)
#     partial_train_targets = np.concatenate(
#         [train_targets[ : i * num_val_samples],
#          train_targets[(i+1) * num_val_samples:]],
#           axis=0)

#     print(partial_train_data.shape)
#     model = build_model()

#     model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=1, verbose=5)

#     val_mse, val_mae = model.evaluate(val_data, val_targets, verbose = 0)
#     all_scores.append(val_mae)

# print(all_scores)
# print(np.mean(all_scores))