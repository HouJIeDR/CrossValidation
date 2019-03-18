#Manual Cross Validation
'''
to evaluate estimator performance/which model has the best performance
'''

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
#Split Train (90%) and Test (10%)
#================================================

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    digits['data'],
    digits['target'],
    test_size = .1
)
# print(len(x_train))
# print(len(x_test))

#================================================
#Model Comparison (LogReg, SVM, RandForest)
#================================================

#Logistic Regression
from sklearn.linear_model import LogisticRegression
modelLR = LogisticRegression(solver = 'lbfgs', multi_class = 'auto')
modelLR.fit(x_train, y_train)
#Testing modelLR Accuracy
print('ModelLR Accuracy = ', modelLR.score(x_test, y_test) * 100, '%')

#SVM
from sklearn.svm import SVC
modelSVC = SVC(gamma = 'auto')
modelSVC.fit(x_train, y_train)
#Testing modelSVC Accuracy
print('ModelSVC Accuracy = ', modelSVC.score(x_test, y_test) * 100, '%')

#Random Forest
from sklearn.ensemble import RandomForestClassifier
modelRF = RandomForestClassifier(n_estimators = 100)
modelRF.fit(x_train, y_train)
#Testing modelRF Accuracy
print('ModelRF Accuracy = ', modelRF.score(x_test, y_test) * 100, '%')

'''
Result e.g.:
ModelLR Accuracy =  94.44444444444444 %
ModelSVC Accuracy =  48.333333333333336 %
ModelRF Accuracy =  97.22222222222221 %

Therefore, 
ModelSVC is the best estimator.
'''