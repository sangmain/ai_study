import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

#데이터 읽어들이기
wine = pd.read_csv("./data/winequality-white.csv", sep=';', encoding='utf-8')

#데이터를 레이블과 데이터로 분리하기
y = wine["quality"]
# x = wine.iloc[:, 0:4]
x = wine.drop(["quality"], axis=1)
# print(x)

#y 레이블 변경하기

newlist = []
for v in list(y):
    if v <= 4:
        newlist += [0]
    elif v <= 7:
        newlist += [1]
    else:
        newlist += [2]
    
y = newlist

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=66)

kfold_cv = KFold(n_splits=5, shuffle=True)

parameters = {"n_estimators": [100, 500, 1000]}

#학습하기
lr = RandomForestClassifier()
clf = RandomizedSearchCV(estimator= lr, param_distributions=parameters, cv = kfold_cv)
clf.fit(x_train, y_train)

print("MODEL ---- RandomForest")

print("최적의 매개변수 = ", clf.best_estimator_)
print("훈련 점수: ", clf.score(x_train, y_train))
print("테스트 점수: ", clf.score(x_test, y_test))
