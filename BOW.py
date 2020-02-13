#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 09:39:24 2020

@author: Yazid BOUNAB
"""

'https://realpython.com/python-keras-text-classification/'

import pickle
import scipy.io as sio
import numpy as np
from sklearn import svm, grid_search
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from PreprocesingTranscriptions import Preprocessing_Dataset

def LoadLabels(Prediction, Annotation, keys):
    Y_A, Y_C, Y_E, Y_I, Y_N, Y_O = [],[],[],[],[],[]
    
    file = open(Annotation, "rb")
    annotation = pickle.load(file, encoding='latin1')
    
    for key in keys:
        if Prediction == 'Inetrview':
           Y_I.append(annotation['interview'][key])
        elif Prediction == 'OCEAN':   
             Y_A.append(annotation['agreeableness'][key])
             Y_C.append(annotation['conscientiousness'][key])
             Y_E.append(annotation['extraversion'][key])
             Y_N.append(annotation['neuroticism'][key])
             Y_O.append(annotation['openness'][key])
    if Prediction == 'Inetrview':
       return Y_I
    elif Prediction == 'OCEAN':     
         return Y_A, Y_C, Y_E, Y_N, Y_O

def LoadDataSet():
    trans_train,tran_val,tran_test, train_keys, val_keys, test_keys = Preprocessing_Dataset()
    
    Y_train = LoadLabels('Inetrview', 'info/annotation_training.pkl', train_keys)
    Y_val = LoadLabels('Inetrview', 'info/annotation_validation.pkl', val_keys)
    Y_test = LoadLabels('Inetrview', 'info/annotation_test.pkl', test_keys)
    
    vectorizer = CountVectorizer()
    vectorizer.fit(trans_train)

    X_train = vectorizer.transform(trans_train)
    X_val = vectorizer.transform(tran_val)
    X_test  = vectorizer.transform(tran_test)

    return  X_train, X_val, X_test, Y_train, Y_val, Y_test
   
def BaseLineModel():
    X_train, X_val, X_test, Y_train, Y_val, Y_test = LoadDataSet()
    #parameters = {'kernel': ('linear', 'rbf','poly'), 'C':[1.5, 10],'gamma': [1e-7, 1e-2],'epsilon':[0.1,0.2,0.5,0.3]}
    #parameters = {'C':[1.5, 1000],'gamma': [1e-9, 1e-1]}

    MODEL = SVR(kernel= 'rbf', C= 1000, gamma= 0.1, epsilon= 0.019)
    MODEL.fit(X_train,Y_train)

    YP_val = MODEL.predict(X_val)
    YP_test = MODEL.predict(X_test)
    
    mae_vl = round(1 - mean_absolute_error(Y_val, YP_val),4)
    mae_ts = round(1 - mean_absolute_error(Y_test, YP_test),4)
    
    mse_vl = round(1 - mean_squared_error(Y_val, YP_val),4)
    mse_ts = round(1 - mean_squared_error(Y_test, YP_test),4)

    r2_vl = round(r2_score(Y_val, YP_val),4)
    r2_ts = round(r2_score(Y_test, YP_test),4)
    
    print('MAE VAL = ' + str(mae_vl))
    print('MAE TS = ' + str(mae_ts))
    print('........................')
    print('MSE VAL = ' + str(mse_vl))
    print('MSE TS = ' + str(mse_ts))
    print('........................')
    print('R2 VAL = ' + str(r2_vl))
    print('R2 TS = ' + str(r2_ts))
    
    filename = 'Interview_model.sav'
    pickle.dump(MODEL, open(filename, 'wb'))
    
#    svr = svm.SVR(kernel= 'rbf', epsilon= 0.04)
#    clf = grid_search.GridSearchCV(svr, parameters)
#    clf.fit(X_train,Y_train)
#    print(clf.best_params_)
    
def LoadBaseLineModel(X_test, Y_test):
    filename = 'Interview_model.sav'
    MODEL = pickle.load(open(filename, 'rb'))
    YP_test = MODEL.predict(X_test, Y_test)
    
    mae_ts = round(1 - mean_absolute_error(Y_test, YP_test),4)
    mse_ts = round(1 - mean_squared_error(Y_test, YP_test),4)
    r2_ts = round(r2_score(Y_test, YP_test),4)

    print('MAE TS = ' + str(mae_ts))
    print('........................')
    print('MSE TS = ' + str(mse_ts))
    print('........................')
    print('R2 TS = ' + str(r2_ts))

BaseLineModel()
