from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

cancer = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify = cancer.target, random_state=42)

# tree = GradientBoostingClassifier(random_state=0)
# tree.fit(x_train, y_train)

# print("훈련세트 정확도: {:.2f}".format(tree.score(x_train, y_train)))
# print("테스트세트 정확도: {:.2f}".format(tree.score(x_test, y_test)))

tree = XGBClassifier(max_depth=1, learning_rate=0.8, n_estimators=2000,
verbosity=1, silent=None,
objective="binary:logistic", booster='gbtree',
n_jobs=-1, nthread=None, gamma=6, min_child_weight=1, max_delta_step=0,
subsample=1, colsample_bytree=1, colsample_bylevel=1,
colsample_bynode=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
base_score=0.5, random_state=0, seed=None, missing=None)
tree.fit(x_train, y_train)
print("훈련세트 정확도: {:.2f}".format(tree.score(x_train, y_train)))
print("테스트세트 정확도: {:.2f}".format(tree.score(x_test, y_test)))

# n_estimators : 클수록 좋다 단점 메모리 많이 차지, 기본값 100
# n_jobs = -1 : cpu 병렬처리
# max_features : 기본값 써라.

print("특성 중요도: \n", tree.feature_importances_)

import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)

    plt.xlabel("특성 중요도")

    plt.ylabel("특성")

    plt.ylim(-1, n_features)

plot_feature_importances_cancer(tree)
# plt.show()