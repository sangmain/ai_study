from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#1. 데이터
learn_data = [[0,0], [1,0], [0,1], [1,1]]
learn_label = [0,1,1,0]

#2. 모델
clf = KNeighborsClassifier(n_neighbors=1)

#3. 학습
clf.fit(learn_data, learn_label)

#4. 평가
x_test = [[0,0], [1,0], [0,1], [1,1]]
y_pred = clf.predict(x_test)

print(x_test, "예측 결과", y_pred)
print('acc = ', accuracy_score([0,1,1,0], y_pred))