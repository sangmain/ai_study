import pandas as pd
import numpy as np

wine = pd.read_csv("./data/winequality-white.csv", sep=';', encoding='utf-8')
wine = np.array(wine)
np.save("wine.npy", wine)
# n = np.load("wine.npy")
# print(n.shape)

pima = np.loadtxt('./data/pima-indians-diabetes.csv', delimiter=',')
pima = np.array(pima)
np.save("pima.npy", pima)
# n = np.load("pima.npy")
# print(n.shape)
iris_data = pd.read_csv("./data/iris.csv", encoding='utf-8')
iris_data = np.array(iris_data)
np.save("iris.npy", iris_data)
# n = np.load("iris.npy")
# print(n.shape)
weather = pd.read_csv('./data/tem10y.csv', encoding='utf-8')
weather = np.array(weather)
np.save("weather.npy", weather)
# n = np.load("weather.npy")
# print(n.shape)

from keras.datasets import cifar10
a = cifar10.load_data()
a = np.array(a)

np.save("cifar.npy", a)
# n = np.load("cifar.npy")
# print(n[0][0].shape)
# print(n[1][0].shape)

from keras.datasets import mnist
a = mnist.load_data()
a = np.array(a)

np.save("mnist.npy", a)
# n = np.load("mnist.npy")
# print(n[0][0].shape)
# print(n[1][0].shape)

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
x = cancer.data
y = cancer.target
y = y[:, None]
a = np.append(x, y, axis=1)
print(a.shape)
np.save("cancer.npy", a)
# n = np.load("cancer.npy")
# print(n[0][30])



from keras.datasets import boston_housing
a = boston_housing.load_data()
a = np.array(a)

np.save("boston.npy", a)
# n = np.load("boston.npy")
# print(n[0][0].shape)
# print(n[1][0].shape)
# print(n)

# np.savetxt("boston.csv", n, delimiter=",")
