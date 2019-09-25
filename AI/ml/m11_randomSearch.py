import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

#붓꽃 데이터 읽어들이기
iris_data = pd.read_csv("./data/iris2.csv", encoding="utf-8")

#붓꽃 데이터를 레이블과 입력 데이터로 분리하기
y = iris_data.loc[:, "Name"]
x = iris_data.loc[:, ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]]

#학습 전용과 테스트 전용 분리하기
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, train_size = 0.8, shuffle=True)

# 그리드 서치에서 사용할 매개변수
# parameters = [
#     {"C": [1, 10, 100, 1000], "kernel":["linear"]},
#     {"C": [1, 10, 100, 1000], "kernel":["rbf"], "gamma":[0.001, 0.0001]},
#     {"C": [1, 10, 100, 1000], "kernel":["sigmoid"], "gamma":[0.001, 0.0001]}
# ]

parameters = {"C": [1, 10, 100, 1000], "kernel":["linear", "rbf", "sigmoid"]}

#그리드 서치
kfold_cv = KFold(n_splits=5, shuffle=True)

from sklearn.model_selection import RandomizedSearchCV
estim = SVC()
clf = RandomizedSearchCV(estimator= estim, param_distributions=parameters, cv = kfold_cv)
clf.fit(x_train, y_train)
print("최적의 매개변수 = ", clf.best_estimator_)

#최적의 매개변수로 평가하기
y_pred = clf.predict(x_test)
print("최종 정답률 = ", accuracy_score(y_test, y_pred))
last_score = clf.score(x_test, y_test)
print("최종 정답률 = ", last_score)
