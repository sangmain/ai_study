import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

#데이터 읽어들이기
wine = pd.read_csv("./data/winequality-white.csv", sep=';', encoding='utf-8')

#데이터를 레이블과 데이터로 분리하기
y = wine["quality"]
# x = wine.iloc[:, 0:4]
x = wine.drop(["quality"], axis=1)
# print(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=66)

#학습하기
model = RandomForestClassifier(n_estimators=1000, class_weight='balanced_subsample')
model.fit(x_train, y_train)
aaa = model.score(x_test, y_test)
#평가하기
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))
print("정답률=", accuracy_score(y_test, y_pred))
print(aaa)