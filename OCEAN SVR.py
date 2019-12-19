#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 11:28:01 2019

@author: polo
"""

import pickle
import numpy as np
import scipy.io as sio

from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_absolute_error

### Data ### 
file = open('Clean_trans_train.pkl', "rb")
Train = pickle.load(file)
file = open('Clean_trans_val.pkl', "rb")
Val = pickle.load(file)
file = open('Clean_trans_test.pkl', "rb")
Test = pickle.load(file)
file = open('Vocabulary.pkl', "rb")
voc = pickle.load(file)

### Annotations ###
file = open('info/annotation_training.pkl', "rb")
A_Train = pickle.load(file, encoding='latin1')
file = open('info/annotation_validation.pkl', "rb")
A_Val = pickle.load(file, encoding='latin1')
file = open('info/annotation_test.pkl', "rb")
A_Test = pickle.load(file, encoding='latin1')

###  ###
def longest(): #44
    List = list(Train.values())
    List.extend(list(Val.values()))
    List.extend(list(Test.values()))
    return max([len(L) for L in List])

def Load_OCEAN_Labels(Clean_Trans,Annotation):
    i = 0
    OCEAN_Labels = []
    X_Features = np.zeros([len(Clean_Trans.keys()),(300*longest())])
    for key in Clean_Trans.keys():
        OCEAN_Labels.append([Annotation['openness'][key], Annotation['conscientiousness'][key],
                             Annotation['extraversion'][key], Annotation['agreeableness'][key], 
                             Annotation['neuroticism'][key]])
        j = 0 
        for token in Clean_Trans[key]:
            if token in voc.keys():
               X_Features[i][j:j+300] = voc[token]
            j = j + 300
        i = i + 1
    
    return X_Features, OCEAN_Labels

def Grid_Search():
    # Tuning of parameters for regression by cross-validation
    K = 5               # Number of cross valiations

    # Parameters for tuning
    parameters = [{'kernel': ['rbf'], 'gamma': [1e-4, 1e-3, 0.01, 0.1, 0.2, 0.5, 0.6, 0.9],'C': [1, 10, 100, 1000, 10000]}]
    print("Tuning hyper-parameters")
    svr = GridSearchCV(SVR(epsilon = 0.01), parameters, cv = K)
    svr.fit(X, y)

    # Checking the score for all parameters
    print("Grid scores on training set:")
    means = svr.cv_results_['mean_test_score']
    stds = svr.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, svr.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"% (mean, std * 2, params))
    
X_TR, OCEAN_TR = Load_OCEAN_Labels(Train,A_Train)
X_VL, OCEAN_VL = Load_OCEAN_Labels(Val,A_Val)
X_TS, OCEAN_TS = Load_OCEAN_Labels(Test,A_Test)

#______________________________________________________________________________
    
    
print('....SVR Learaning....')
    
#MODEL = SVR(kernel= 'rbf', C= 1e4, gamma= 0.2)
MODEL = MultiOutputRegressor(SVR(kernel= 'rbf', C= 1e4, gamma= 0.2))

MODEL.fit(X_TR,OCEAN_TR)


P_OCEAN_VL = MODEL.predict(X_VL)
P_OCEAN_TS = MODEL.predict(X_TS)

mae_vl = 1 - mean_absolute_error(OCEAN_VL, P_OCEAN_VL)
mae_ts = 1 - mean_absolute_error(OCEAN_TS, P_OCEAN_TS)

print('MAE VAL = ' + str(mae_vl))
print('MAE TS = ' + str(mae_ts))

pickle.dump(MODEL, open('First_Impression_Model.pkl', 'wb'))