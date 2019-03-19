#K-Fold Cross Validation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#================================================
#Load Data
#================================================

from sklearn.datasets import load_digits
digits = load_digits()
# print(dir(digits))

#================================================
#Machine Learning Model
#================================================

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

#================================================
#Stratified K-Fold Cross Validation
#================================================

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
k = StratifiedKFold(n_splits = 100)

#K-Fold
for train_index, test_index in k.split(digits['data'], digits['target']):
    x_train = digits['data'][train_index]
    y_train = digits['target'][train_index]

#Testing Cross Validation Score
print(cross_val_score(
    LogisticRegression(solver = 'lbfgs', multi_class = 'auto'),
    x_train,
    y_train
).mean())

print(cross_val_score(
    SVC(gamma = 'auto'),
    x_train,
    y_train
).mean())

print(cross_val_score(
    RandomForestClassifier(n_estimators = 100),
    x_train,
    y_train
).mean())