from keras.models import Sequential
from keras.layers import Dense
import numpy
import tensorflow as tf
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score


#시드값 생성
seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

#데이터 로드
dataset = numpy.loadtxt('./data/pima-indians-diabetes.csv', delimiter=',')
x = dataset[:, 0:8]
y = dataset[:, 8]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, test_size=0.2)


from sklearn.preprocessing import MinMaxScaler, StandardScaler
# scaler = StandardScaler()
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# x_test = dataset[:, 8:]
#모델의 생성
# model = SVC()
model = LinearSVC()
# model = KNeighborsClassifier(n_neighbors=4)
# model = KNeighborsRegressor(n_neighbors=1)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)
#결과 출력
# print(x_test, "예측 결과", y_pred)
# print("\n Accuracy: %.4f" % (model.evaluate(x,y)[1]))
print('acc = ', accuracy_score(y_test, y_pred))
# print('acc = ', accuracy_score([0,1,1,0], y_pred))
