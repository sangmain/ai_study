import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.utils.testing import all_estimators
import warnings
warnings.filterwarnings('ignore')

#붓꽃 데이터 읽어들이기
iris_data = pd.read_csv("./data/iris2.csv", encoding="utf-8")

#붓꽃 데이터를 레이블과 입력 데이터로 분리하기
y = iris_data.loc[:, "Name"]
x = iris_data.loc[:, ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]]

#학습 전용과 테스트 전용 분리하기
warnings.filterwarnings("ignore")
#K-분할 크로스 발리데이션 전용 객체

for i in range(3, 10):

    fold_num = i
    k_fold_cv = KFold(n_splits=fold_num, shuffle=True)

    #classifier 알고리즘 모두 추출하기 --- (*1)
    warnings.filterwarnings("ignore")
    # allAlgorithms = all_estimators(type_filter="classifier")
    allAlgorithms = all_estimators(type_filter="classifier")
    # print(allAlgorithms)
    # print(len(allAlgorithms))

    import numpy as np

    highest = 0.0
    names = ""
    score_a = None

    for(name, algorithm) in allAlgorithms:
        # 각 알고리즘 객체 생성하기 --- (*2)
        clf = algorithm()

        #score 멧드를 가진 클래스를 대상으로 하기
        if hasattr(clf, "score"):
            #크로스 밸리데이션
            scores = cross_val_score(clf, x, y, cv=k_fold_cv)
            # print(name, "의 정답률")
            sums = 0.0
            avg = 0.0
            for i in range(fold_num):
                sums = sums + scores[i]

            avg = sums / fold_num

            if highest < avg:
                names = name
                highest = avg
                score_a = scores

    print(fold_num, names, score_a)