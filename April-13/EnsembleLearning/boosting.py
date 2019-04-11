#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 21:53:27 2019

@author: rahmeen
"""

from sklearn import model_selection
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier  # Boosting Algorithm
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

import numpy as np


#Load data 

loadedData = load_breast_cancer()
X = loadedData.data
Y = loadedData.target

#Split data in training and testing set 
X_fit, X_eval, y_fit, y_test= model_selection.train_test_split( X, Y, test_size=0.20, random_state=1 )

#Define a decision tree classifier
cart = DecisionTreeClassifier()
num_trees = 25

#Create classification model for bagging
ada_boost = AdaBoostClassifier(base_estimator=cart, n_estimators=num_trees, learning_rate = 0.1)
grad_boost = GradientBoostingClassifier()
xgb_boost = XGBClassifier()
#Train Classification model
ada_boost.fit(X_fit, y_fit)
grad_boost.fit(X_fit, y_fit)
xgb_boost.fit(X_fit, y_fit)

#Test trained model over test set
pred_label1 = ada_boost.predict(X_eval)
nnz = np.float(np.shape(y_test)[0] - np.count_nonzero(pred_label1 - y_test))
acc = 100*nnz/np.shape(y_test)[0]

#Print accuracy of the model
print('accuracy of Ada Boost is: '+str(acc))

#Test trained model over test set
pred_label2 = grad_boost.predict(X_eval)
nnz = np.float(np.shape(y_test)[0] - np.count_nonzero(pred_label2 - y_test))
acc = 100*nnz/np.shape(y_test)[0]

#Print accuracy of the model
print('accuracy of Grad boost is: '+str(acc))

#Test trained model over test set
pred_label3 = xgb_boost.predict(X_eval)
nnz = np.float(np.shape(y_test)[0] - np.count_nonzero(pred_label3 - y_test))
acc = 100*nnz/np.shape(y_test)[0]

#Print accuracy of the model
print('accuracy of XGBoost is: '+str(acc))
