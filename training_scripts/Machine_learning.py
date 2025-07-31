# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 02:21:55 2025

@author: Pradnya Kamble

"""
# Importing required librariesimport os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score, 
    confusion_matrix, roc_curve, auc, precision_recall_curve, classification_report
)
from sklearn.ensemble import RandomForestClassifier
import joblib
from openpyxl import Workbook
from scipy.stats import sem
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# Load the training and test data
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
Y_train = pd.read_csv('Y_train.csv')
Y_test = pd.read_csv('Y_test.csv')

path_results_op = 'path_to_output_dir'

# KNN
param_grid_knn = {'n_neighbors': [3,5,...],
              'p': [1, 2]}  

knn_classifier = KNeighborsClassifier()

grid_knn = GridSearchCV(estimator=knn_classifier, param_grid=param_grid_knn, n_jobs=-1,cv=10, verbose=2, scoring='accuracy')

grid_knn.fit(X_train, Y_train)

best_model_knn = grid_knn.best_estimator_

test_predictions_knn = best_model_knn.predict(X_test)
train_predictions_knn = best_model_knn.predict(X_train)

# Train and Test Predictions
train_predictions_knn = best_model_knn.predict(X_train)
test_predictions_knn = best_model_knn.predict(X_test)

# Train and Test Metrics
train_accuracy_knn = accuracy_score(Y_train, train_predictions_knn)
test_accuracy_knn = accuracy_score(Y_test, test_predictions_knn)

# LR

param_grid_LR = {
    'C': [1.0, 0.1, 0.01],  
    'max_iter': [100, 200],  
    'l1_ratio': [0, 0.1]
}

LR_classifier = LogisticRegression()

grid_LR = GridSearchCV(LR_classifier, param_grid_LR, cv=10, verbose=1, n_jobs=-1, scoring='accuracy') 

grid_LR.fit(X_train, Y_train)

best_model_LR = grid_LR.best_estimator_

test_predictions_LR = best_model_LR.predict(X_test)
train_predictions_LR = best_model_LR.predict(X_train)

# Random Forest

rf_classifier = RandomForestClassifier(random_state=42)

param_grid_rf = {
    'n_estimators': [50, 100, 150],
    'max_features': [2,5,10,15],
    'max_depth': [6, 8],
    'min_samples_split': [2, 5]}

grid_rf = GridSearchCV(estimator=rf_classifier, param_grid=param_grid_rf, cv=10, n_jobs=-1, verbose=1, scoring='accuracy')

grid_rf.fit(X_train, Y_train)
best_model_rf = grid_rf.best_estimator_

test_predictions_rf = best_model_rf.predict(X_test)

train_predictions_rf = best_model_rf.predict(X_train)

# Train and Test Predictions
train_predictions_rf = best_model_rf.predict(X_train)
test_predictions_rf = best_model_rf.predict(X_test)

# Train and Test Metrics
train_accuracy_rf = accuracy_score(Y_train, train_predictions_rf)
test_accuracy_rf = accuracy_score(Y_test, test_predictions_rf)

# XGBoost
xgb_classifier = XGBClassifier(random_state=42)
# A parameter grid for XGBoost
param_grid_xgb = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1],
        'subsample': [0.6, 0.8, 1.0],
        'max_depth': [3, 4]
        }
grid_xgb = GridSearchCV(estimator=xgb_classifier, param_grid=param_grid_xgb, cv=10, n_jobs=-1, verbose=1, scoring='accuracy')

grid_xgb.fit(X_train, Y_train)
print("Best Parameters: ", grid_xgb.best_params_)

best_model_xgb = grid_xgb.best_estimator_
test_predictions_xgb = best_model_xgb.predict(X_test)
train_predictions_xgb = best_model_xgb.predict(X_train)

# Train and Test Predictions
train_predictions_xgb = best_model_xgb.predict(X_train)
test_predictions_xgb = best_model_xgb.predict(X_test)

# Train and Test Metrics
train_accuracy_xgb = accuracy_score(Y_train, train_predictions_xgb)
test_accuracy_xgb = accuracy_score(Y_test, test_predictions_xgb)

# SVM
param_grid_svm = {'C': [0.1, 0.5, 1],  #0.01, 0.05, 1.5, 2, 2.5, 3
              'gamma': [ 0.001, 0.01], 
              'kernel': ['linear','rbf'] , 
              'degree': [1,2,3]}  

svm_model = SVC(probability=True)

grid_svm = GridSearchCV(estimator=svm_model, param_grid=param_grid_svm, refit = True, verbose = 1, cv=10, n_jobs=-1) 

grid_svm.fit(X_train, Y_train)
best_model_svm = grid_svm.best_estimator_

test_predictions_svm = best_model_svm.predict(X_test)
train_predictions_svm = best_model_svm.predict(X_train)

# Train and Test Predictions
train_predictions_svm = best_model_svm.predict(X_train)
test_predictions_svm = best_model_svm.predict(X_test)

# Train and Test Metrics
train_accuracy_svm = accuracy_score(Y_train, train_predictions_svm)
test_accuracy_svm = accuracy_score(Y_test, test_predictions_svm)























