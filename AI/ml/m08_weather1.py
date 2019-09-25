from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

#직선 회귀 분석하기
lr = LinearRegression(normalize=True)
lr.fit(x_train, y_train) #학습하기
y_pred = lr.predict(x_test) #예측

print(lr.score(x_test, y_test))
#결과를 그래프로 그리기
plt.figure(figsize=(10, 6), dpi=100)
plt.plot(y_test, c='r')
plt.plot(y_pred, c='b')
plt.savefig('tenki-kion-lr.png')
plt.show()