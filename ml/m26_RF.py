from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor


#기온 데이터 읽어들이기
df = pd.read_csv('./data/tem10y.csv', encoding='utf-8')

#데이터를 학습 전용과 테스트 전용으로 분리하기
train_year = (df['연'] <= 2015)
test_year = (df["연"] >= 2016)
interval = 6

#과거 6명의 데이터를 기반으로 학습할 데이터 만들기
def make_data(data):
    x = []
    y = []
    temps = list(data["기온"])
    for i in range(len(temps)):
        if i < interval: continue
        y.append(temps[i])
        xa = []

        for p in range(interval):
            d = i + p - interval
            xa.append(temps[d])
        x.append(xa)
    return (x, y)

x_train, y_train =  make_data(df[train_year])
x_test, y_test = make_data(df[test_year])

kfold_cv = KFold(n_splits=5, shuffle=True)



parameters = {"n_estimators": [100, 500, 1000]}
    # {"C": [1, 10, 100, 1000], "kernel":["rbf"], "gamma":[0.001, 0.0001]},
    # {"C": [1, 10, 100, 1000]}

#직선 회귀 분석하기
lr = RandomForestRegressor()
clf = RandomizedSearchCV(estimator= lr, param_distributions=parameters, cv = kfold_cv)
clf.fit(x_train, y_train) #학습하기

print("MODEL ---- RandomForest")

print("최적의 매개변수 = ", clf.best_estimator_)
print("훈련 점수: ", clf.score(x_train, y_train))
print("테스트 점수: ", clf.score(x_test, y_test))

# from sklearn.metrics import r2_score
# r2_y_predict = r2_score(y_test, y_predict)
# print("R2: ", r2_y_predict)
