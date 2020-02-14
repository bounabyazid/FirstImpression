#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 09:39:24 2020

@author: Yazid BOUNAB
"""

'https://realpython.com/python-keras-text-classification/'

'https://stats.stackexchange.com/questions/291297/countvectorizer-as-n-gram-presence-and-count-feature'
'https://stackoverflow.com/questions/28894756/countvectorizer-does-not-print-vocabulary/44320484'

import pickle
import scipy.io as sio
import numpy as np

from keras import layers
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from PreprocesingTranscriptions import Preprocessing_Dataset

import matplotlib.pyplot as plt
plt.style.use('ggplot')

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
    
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(trans_train)

    X_train = tokenizer.texts_to_sequences(trans_train)
    X_val = tokenizer.texts_to_sequences(tran_val)
    X_test = tokenizer.texts_to_sequences(tran_test)
    
    maxlen = 100

    X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
    X_val = pad_sequences(X_val, padding='post', maxlen=maxlen)
    X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

    return  X_train, X_val, X_test, Y_train, Y_val, Y_test
   
def KerasDeeepLearning():
    X_train, X_val, X_test, Y_train, Y_val, Y_test = LoadDataSet()
    
    input_dim = X_train.shape[1]  # Number of features

    MODEL = Sequential()
#    MODEL.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
#    MODEL.add(layers.Dense(1, activation='sigmoid'))

    MODEL.add(layers.Dense(40, input_dim=input_dim, activation='relu'))
    MODEL.add(layers.Dense(15, activation='relu'))
    MODEL.add(layers.Dense(1, activation='sigmoid'))

#    MODEL.add(layers.Dense(30, input_dim=input_dim, activation='relu'))
#    MODEL.add(layers.Dense(15, activation='relu'))
#    MODEL.add(layers.Dense(7, activation='relu'))
#    MODEL.add(layers.Dense(1, activation='sigmoid'))


    MODEL.compile(loss='mse', optimizer='adam', metrics=['mae'])
    MODEL.summary()

    History = MODEL.fit(X_train, Y_train, epochs=25, verbose=False, 
                        validation_data=(X_val, Y_val), batch_size=30)
    
    loss, MEA = MODEL.evaluate(X_train, Y_train, verbose=False)
    print("Training MAE: {:.4f} Correct {:.4f}".format(MEA,1-MEA))
    
    loss, MEA = MODEL.evaluate(X_test, Y_test, verbose=False)
    print("Testing MAE:  {:.4f} Correct {:.4f}".format(MEA,1-MEA))

    return History

#    filename = 'Interview_DL_model.sav'
#    pickle.dump(MODEL, open(filename, 'wb'))

def plot_history(History):
    mae = History.history['mean_absolute_error']
    val_mae = History.history['val_mean_absolute_error']
    loss = History.history['loss']
    val_loss = History.history['val_loss']
    x = range(1, len(mae) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, mae, 'b', label='Training mean absolute error')
    plt.plot(x, val_mae, 'r', label='Validation mean absolute error')
    plt.title('Training and validation mae')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
    
def LoadBaseLineModel(X_test, Y_test):
    filename = 'Interview_DL_model.sav'
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

History = KerasDeeepLearning()
plot_history(History)
