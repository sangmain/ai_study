import numpy as np
from sklearn.model_selection import StratifiedKFold
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4],[3,3], [5,5]])
y = np.array([0, 0, 1, 1, 5, 6])
skf = StratifiedKFold(n_splits=2)
skf.get_n_splits(X, y)

# print(skf)  
print("original X: ", X)

for train_index, test_index in skf.split(X, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print("X_train: ",X_train)
    print("X_test: ", X_test)
    print()
