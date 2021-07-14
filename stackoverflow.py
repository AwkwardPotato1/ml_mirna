#imports

from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.under_sampling import NearMiss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV




#getting data

url = "https://drive.google.com/file/d/17vNpOppKGP6jev9fk4GIoeP0mcefyY7C/view?usp=sharing"
path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
df = pd.read_csv(path)

#Separating features from labels
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

#Scaling
sc = StandardScaler()
X = sc.fit_transform(X)

#Splitting the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
#
#Imbalance handling
sm = SMOTE(random_state=2)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train.ravel())

nr = NearMiss()
X_train_miss, y_train_miss = nr.fit_resample(X_train, y_train.ravel())
#
#Model Parameter tuning
model_params = {
    'svm': {
        'model': svm.SVC(gamma='auto'),
        'params': {
            'C': [0.001, 0.01, 0.1, 0.5, 1, 5, 6, 7, 10, 20, 40],
            'kernel': ['rbf', 'linear']
        }
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params': {
            'n_estimators': [10, 20, 30, 40, 60, 80, 90, 100, 130, 150, 200, 250, 270, 300, 400]
        }
    },
     'logistic_regression' : {
        'model': LogisticRegression(multi_class='auto'),
        'params': {
            'C': [0.01, 0.1, 0.5, 1, 3, 5, 10, 15, 20],
            'solver' : ['liblinear','sag','saga','newton-cg']
        }
    },
    'knn': {
        'model': KNeighborsClassifier(),
        'params': {
            'n_neighbors': [1, 2, 3, 5, 7, 10, 12, 20, 50, 70, 100],
            'metric': ['euclidean', 'manhattan']
        }
    },
}
#
#Gridsearch for Near-Miss
scores = []
for model_name, mp in model_params.items():
    clf = GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False, n_jobs=-1)
    clf.fit(X_train_miss, y_train_miss)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })

df_nm = pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])
#
#Gridsearch for SMOTE
scores1 = []
for model_name, mp in model_params.items():
    clf1 = GridSearchCV(mp['model'], mp['params'], cv=10, return_train_score=False, n_jobs=-1)   #cv = 10 since it's a larger dataset
    clf1.fit(X_train_sm, y_train_sm)
    scores1.append({
        'model': model_name,
        'best_score': clf1.best_score_,
        'best_params': clf1.best_params_
    })

df_sm = pd.DataFrame(scores1, columns=['model', 'best_score', 'best_params'])

grid_results = pd.concat([df_sm, df_nm])
#
#Getting the best parameters to train the classifiers

#SMOTE
clf_svm_sm = SVC(kernel= 'rbf', probability=True, C=7)
clf_rf_sm = RandomForestClassifier(verbose=True, n_estimators=100, n_jobs=-1)
clf_lg_sm = LogisticRegression(solver='newton-cg', multi_class='auto', verbose=True, C= 100, n_jobs=-1)
clf_knn_sm = KNeighborsClassifier(n_neighbors=1, metric="manhattan", n_jobs=-1)

# NM
clf_svm_nm = SVC(kernel= 'rbf', probability=True, C=7)
clf_rf_nm = RandomForestClassifier(verbose=True, n_estimators=10, n_jobs=-1)
clf_lg_nm = LogisticRegression(solver='newton-cg', multi_class='auto', verbose=True, C= 15, n_jobs=-1)
clf_knn_nm = KNeighborsClassifier(n_neighbors=1, metric="manhattan", n_jobs=-1)
#
#Fitting
#SMOTE
clf_svm_sm.fit(X_train_sm, y_train_sm)
clf_rf_sm.fit(X_train_sm, y_train_sm)
clf_lg_sm.fit(X_train_sm, y_train_sm)
clf_knn_sm.fit(X_train_sm, y_train_sm)
#
#Near-Miss
clf_svm_nm.fit(X_train_miss, y_train_miss)
clf_rf_nm.fit(X_train_miss, y_train_miss)
clf_lg_nm.fit(X_train_miss, y_train_miss)
clf_knn_nm.fit(X_train_miss, y_train_miss)
#
#Predictions
y_pred_svm_sm = clf_svm_sm.predict(X_test)
y_pred_rf_sm = clf_rf_sm.predict(X_test)
y_pred_lg_sm = clf_lg_sm.predict(X_test)
y_pred_knn_sm = clf_knn_sm.predict(X_test)

y_pred_svm_nm = clf_svm_nm.predict(X_test)
y_pred_rf_nm = clf_rf_nm.predict(X_test)
y_pred_lg_nm = clf_lg_nm.predict(X_test)
y_pred_knn_nm = clf_knn_nm.predict(X_test)
#
#Getting confusion matrix for Random forest and SVM for example
coX1 = confusion_matrix(y_test, y_pred_svm_sm)
coX2 = confusion_matrix(y_test, y_pred_rf_sm)
coX3 = confusion_matrix(y_test, y_pred_svm_nm)
coX4 = confusion_matrix(y_test, y_pred_rf_nm)

print("<--Performance on test data-->","\nConfusion Matrix for SVM SMOTE:\n", coX1 , "\nConfusion Matrix for Random Forest SMOTE:\n", coX2 ,"\nConfusion Matrix for SVM Near-Miss:\n", coX3 ,"\nConfusion Matrix for Random Forest Near-Miss:\n", coX4)
#
##Validation Using Independent Test Dataset
#
#Getting test data and pre-processing
url_0 = "https://drive.google.com/file/d/1YCC24nBq98LCZwkAyc7RmHzfMP0pC7zg/view?usp=sharing"
path = 'https://drive.google.com/uc?export=download&id='+url_0.split('/')[-2]
df_test = pd.read_csv(path)

#Separating features from labels
X1 = df_test.iloc[:, :-1].values
y1 = df_test.iloc[:, -1].values

#Scaling
X1 = sc.fit_transform(X1)

#Predictions
y_pred_svm_sm_test = clf_svm_sm.predict(X1)
y_pred_rf_sm_test = clf_rf_sm.predict(X1)
y_pred_svm_nm_test = clf_svm_nm.predict(X1)
y_pred_rf_nm_test = clf_rf_nm.predict(X1)

#Getting confusion matrix for Random forest and SVM for example
coX1t = confusion_matrix(y1, y_pred_svm_sm_test)
coX2t = confusion_matrix(y1, y_pred_rf_sm_test)
coX3t = confusion_matrix(y1, y_pred_svm_nm_test)
coX4t = confusion_matrix(y1, y_pred_rf_nm_test)

print("<--Performance on INDEPENDENT test data for generalisation-->","\nConfusion Matrix for SVM SMOTE:\n", coX1t , "\nConfusion Matrix for Random Forest SMOTE:\n", coX2t ,"\nConfusion Matrix for SVM Near-Miss:\n", coX3t ,"\nConfusion Matrix for Random Forest Near-Miss:\n", coX4t)
#
#Testing only with positive examples from the independent set
df_test_P = df_test[ df_test["SEQ"] == 1]

#Separating features from labels
XP = df_test_P.iloc[:, :-1].values
yP = df_test_P.iloc[:, -1].values

#Scaling
XP = sc.fit_transform(XP)

#Predictions
y_pred_svm_sm_testP = clf_svm_sm.predict(XP)
y_pred_rf_sm_testP = clf_rf_sm.predict(XP)
y_pred_svm_nm_testP = clf_svm_nm.predict(XP)
y_pred_rf_nm_testP = clf_rf_nm.predict(XP)

#Getting confusion matrix for Random forest and SVM for example
coX1p = confusion_matrix(yP, y_pred_svm_sm_testP)
coX2p = confusion_matrix(yP, y_pred_rf_sm_testP)
coX3p = confusion_matrix(yP, y_pred_svm_nm_testP)
coX4p = confusion_matrix(yP, y_pred_rf_nm_testP)

print("<--Performance of ONLY POSITIVE sets from same INDEPENDENT data-->","\nConfusion Matrix for SVM SMOTE:\n", coX1p , "\nConfusion Matrix for Random Forest SMOTE:\n", coX2p ,"\nConfusion Matrix for SVM Near-Miss:\n", coX3p ,"\nConfusion Matrix for Random Forest Near-Miss:\n", coX4p)
#

