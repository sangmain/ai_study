import pandas as pd
import numpy as np

# Load data
train = pd.read_csv('data/train.csv', nrows=20000)
test = pd.read_csv('data/test.csv', nrows=200000)

print(train.shape)
print(test.shape)

# Lets Drop What not requeired .. In this case Both have id which need to be removed

y_train = train['target']
test_id = test['id'] # Future Requirement
train.drop(['id', 'target'], axis=1, inplace=True)
test.drop(['id'], axis=1, inplace=True)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train, y_train, test_size=0.33, stratify=y_train)

from sklearn.preprocessing import LabelEncoder

# Auto encodes any dataframe column of type category or object.
def dummyEncode(df):
    s1 = [ 'nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4', 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5']
    
    for feature in s1:
        dummy = pd.get_dummies(df['{}'.format(feature)])
        df = pd.concat([df, dummy], axis=1)
        
    return df
    

X_train = dummyEncode(X_train)
X_test = dummyEncode(X_test)

TF_Map = { 'T' : 1, 'F' : 0}
YN_Map = { 'Y' : 1, 'N' : 0}


X_train['bin_3_'] =  X_train['bin_3'].map(TF_Map)
X_train['bin_4_'] = X_train['bin_4'].map(YN_Map)

X_test['bin_3_'] = X_test['bin_3'].map(TF_Map)
X_test['bin_4_'] = X_test['bin_4'].map(YN_Map)

def dropExtrafeatures(df):
    df.drop([ 'bin_4', 'bin_3', 'nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4', 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5'], axis=1, inplace=True)
    return df

X_train = dropExtrafeatures(X_train)
X_test = dropExtrafeatures(X_test)


print(X_train.shape)
print(X_test.shape)

# Get missing columns in the training test
missing_cols = set( X_train.columns ) - set( X_test.columns )
# Add a missing column in test set with default value equal to 0
for c in missing_cols:
    X_test[c] = 0
# Ensure the order of column in the test set is in the same order than in train set
X_test = X_test[X_train.columns]

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt

def plotROCCurveGraph(X_train_roc, y_train_roc, X_test_roc, y_test_roc, best_alpha):
    # for i in tqdm(parameters):
    neigh = LogisticRegression( C=best_alpha, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='warn', n_jobs=None, penalty='l2',
                   random_state=None, solver='warn', tol=0.0001, verbose=0,
                   warm_start=False)
    neigh.fit(X_train_roc, y_train_roc)

    y_train_pred = neigh.predict_proba(X_train_roc)[:,1]
    y_test_pred = neigh.predict_proba(X_test_roc)[:,1]

    train_fpr, train_tpr, tr_thresholds = roc_curve(y_train_roc, y_train_pred)
    test_fpr, test_tpr, te_thresholds = roc_curve(y_test_roc, y_test_pred)

    m_Auc = str(auc(train_fpr, train_tpr))


    plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))
    plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))
    plt.legend()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.grid()
    plt.show()
    
    
    
    return neigh, m_Auc,


clf, m1_Auc, = plotROCCurveGraph(X_train, y_train, X_test, y_test, 0.1)

test = dummyEncode(test)

TF_Map = { 'T' : 1, 'F' : 0}
YN_Map = { 'Y' : 1, 'N' : 0}


test['bin_3_'] =  test['bin_3'].map(TF_Map)
test['bin_4_'] = test['bin_4'].map(YN_Map)

test = dropExtrafeatures(test)

missing_cols = set( X_train.columns ) - set( test.columns )
# Add a missing column in test set with default value equal to 0
for c in missing_cols:
    test[c] = 0
# Ensure the order of column in the test set is in the same order than in train set
test = test[X_train.columns]

def index_marks(nrows, chunk_size):
    return range(1 * chunk_size, (nrows // chunk_size + 1) * chunk_size, chunk_size)

def split(dfm, chunk_size):
    indices = index_marks(dfm.shape[0], chunk_size)
    return np.split(dfm, indices)

chunks = split(test, 10000)

result_toSubmit = []

for c in chunks:
    if (c.shape[0] != 0):
        result_toSubmit.extend(clf.predict(c))
        print("Shape: {}; {}".format(c.shape, c.index))

submission = pd.DataFrame({'id': test_id, 'target': result_toSubmit})
submission.to_csv('v2_submission.csv', index=False)