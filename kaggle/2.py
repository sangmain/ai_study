import numpy as np # linear algebra
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier
# import xgboost as xgb

## Hyperopt modules
# from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK, STATUS_RUNNING
# from functools import partial

# Load data
df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')
submission = pd.read_csv('data/sample_submission.csv', index_col='id')


y = df_train['target']
df_train.drop(['id', 'target'], axis=1, inplace=True)
test_id = df_test['id'] # Future Requirement

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df_train, y, test_size=0.33, stratify=y)

def dummyEncode(df):
    s1 = [ 'nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4', 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5']
    
    for feature in s1:
        dummy = pd.get_dummies(df['{}'.format(feature)])
        df = pd.concat([df, dummy], axis=1)
        
    return df
    

x_train = dummyEncode(x_train)
x_test = dummyEncode(x_test)
print("Hello")
# x_test = np.load('x_test.npy')
print(x_test)
def preprocessing(x_temp):
        
    #bin_3, bin_4 전처리 mapping
    TF_Map = {'T': 1, 'F': 0}
    YN_Map = {'Y': 1, 'N': 0}
    x_temp['bin_3'] = x_temp['bin_3'].map(TF_Map)
    x_temp['bin_4'] = x_temp['bin_4'].map(YN_Map)


    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(x_temp['nom_0'])
    x_temp['nom_0'] = le.transform(x_temp['nom_0'])
    le.fit(x_temp['nom_1'])
    x_temp['nom_1'] = le.transform(x_temp['nom_1'])
    le.fit(x_temp['nom_2'])
    x_temp['nom_2'] = le.transform(x_temp['nom_2'])
    le.fit(x_temp['nom_3'])
    x_temp['nom_3'] = le.transform(x_temp['nom_3'])
    le.fit(x_temp['nom_4'])
    x_temp['nom_4'] = le.transform(x_temp['nom_4'])
    
    return x_temp.copy()




x_train = preprocessing(x_train.copy())
x_test = preprocessing(x_test.copy())

#전처리하지않은부분 제거
def dropExtrafeatures(df):
    df.drop(['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5'], axis=1, inplace=True)
    return df


x_train = dropExtrafeatures(x_train)

x_test = dropExtrafeatures(x_test)
print(x_train)

# 특성 중요도
tree = DecisionTreeClassifier(random_state=0)
tree.fit(x_train, y_train)
print("특성 중요도: \n", tree.feature_importances_)


#모델 구성
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


##### 0.6526
clf = XGBClassifier(
            n_estimators=500, random_state=4, 
            verbose=True, 
            tree_method='gpu_hist')

## 0.6252
# clf = LogisticRegression( C=0.1, class_weight=None, dual=False, fit_intercept=True,
#                 intercept_scaling=1, l1_ratio=None, max_iter=100,
#                 multi_class='warn', n_jobs=None, penalty='l2',
#                 random_state=None, solver='warn', tol=0.0001, verbose=0,
#                 warm_start=False)

clf.fit(x_train, y_train)
print("finished")

y_train_pred = clf.predict_proba(x_train)[:,1]
y_test_pred = clf.predict_proba(x_test)[:,1]

train_fpr, train_tpr, tr_thresholds = roc_curve(y_train, y_train_pred)
test_fpr, test_tpr, te_thresholds = roc_curve(y_test, y_test_pred)

m_Auc = str(auc(train_fpr, train_tpr))


plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))
plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))
plt.legend()
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curve")
plt.grid()
plt.show()

clf.save_model("model.h5")
#############################################
#### Test

# del x_train,
# df_test = dummyEncode(df_test)

# df_test = preprocessing(df_test)

# df_test = dropExtrafeatures(df_test)

# missing_cols = set( x_train.columns ) - set( df_test.columns )
# # Add a missing column in test set with default value equal to 0
# for c in missing_cols:
#     df_test[c] = 0
# # Ensure the order of column in the test set is in the same order than in train set
# df_test = df_test[x_train.columns]

# def index_marks(nrows, chunk_size):
#     return range(1 * chunk_size, (nrows // chunk_size + 1) * chunk_size, chunk_size)

# def split(dfm, chunk_size):
#     indices = index_marks(dfm.shape[0], chunk_size)
#     return np.split(dfm, indices)

# chunks = split(df_test, 10000)

# result_toSubmit = []

# for c in chunks:
#     if (c.shape[0] != 0):
#         result_toSubmit.extend(clf.predict(c))
#         print("Shape: {}; {}".format(c.shape, c.index))

# submission = pd.DataFrame({'id': test_id, 'target': result_toSubmit})
# submission.to_csv('self_submission.csv', index=False)
