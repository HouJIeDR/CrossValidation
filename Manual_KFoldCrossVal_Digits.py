#K-Fold Cross Validation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#================================================
#Load Data
#================================================

from sklearn.datasets import load_digits
digits = load_digits()

#================================================
#Split Train (90%) and Test (10%)
#================================================

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    digits['data'],
    digits['target'],
    test_size = .1,
)

#================================================
#Manual K-Fold Cross Validation
#================================================

#Machine Learning Model
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

#Function Get Score Manually
def get_score(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    return model.score(x_test, y_test)

# print(get_score(LogisticRegression(solver = 'lbfgs', multi_class = 'auto'), x_train, x_test, y_train, y_test))
# print(get_score(SVC(gamma = 'auto'), x_train, x_test, y_train, y_test))
# print(get_score(RandomForestClassifier(n_estimators = 100), x_train, x_test, y_train, y_test))

#Import Sklearn K-Fold
from sklearn.model_selection import KFold
k = KFold(n_splits = 3)           #k max: n data in dataset
 
scoreLR = []
scoreSVM = []
scoreRF = []

for train_index, test_index in k.split(digits['data']):
    x_train = digits['data'][train_index]
    x_test = digits['data'][test_index]
    y_train = digits['target'][train_index]
    y_test = digits['target'][test_index]

    scoreLR.append(get_score(LogisticRegression(solver = 'lbfgs', multi_class = 'auto'), x_train, x_test, y_train, y_test))
    scoreSVM.append(get_score(SVC(gamma = 'auto'), x_train, x_test, y_train, y_test))
    scoreRF.append(get_score(RandomForestClassifier(n_estimators = 100), x_train, x_test, y_train, y_test))

print(scoreLR)
print(scoreSVM)
print(scoreRF)

#Print Mean Score
print(np.array(scoreLR).mean())
print(np.array(scoreSVM).mean())
print(np.array(scoreRF).mean())