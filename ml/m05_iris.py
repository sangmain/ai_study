import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score  

#붓꽃데이터 읽어들이기
colnames = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Name'] 
iris_data = pd.read_csv("./data/iris.csv", names= colnames, encoding='utf-8')

#붓꽃 데이터를 레이블과 입력 데이터로 뷴리하기
y = iris_data.loc[:, "Name"]
x = iris_data.loc[:, ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]]

#학습 전용과 테스트 전용 분리하기
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, train_size = 0.8, shuffle=True)

#학습하기
# clf = SVC() #96.6
clf = LinearSVC() # 83.3
# clf = KNeighborsClassifier(n_neighbors=1) #랜덤
clf.fit(x_train, y_train)

#평가하기
y_pred = clf.predict(x_test)
print("정답률: ", accuracy_score(y_test, y_pred))