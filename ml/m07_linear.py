from sklearn.datasets import load_boston
import numpy as np

boston = load_boston()
print(boston.data.shape)
print(boston.keys())
print(boston.target)
print(boston.target.shape)

x = boston.data
y = boston.target


print(type(boston))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, test_size = 0.2)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
# lasso = Lasso().fit(x, y)
# ridge = Ridge()
ridge = Ridge(alpha=0.05, normalize=True)
ridge.fit(x_train, y_train)
print(ridge.score(x_test, y_test))



