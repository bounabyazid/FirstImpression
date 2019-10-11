import pickle
import scipy.io as sio
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error


###  ###
def longest():
    List = list(train.values())
    List.extend(list(val.values()))
    List.extend(list(test.values()))
    
    return max([len(L) for L in List])

### Data ### 
file = open('Clean_trans_train.pkl', "rb")
train = pickle.load(file)
file = open('Clean_trans_val.pkl', "rb")
val = pickle.load(file)
file = open('Clean_trans_test.pkl', "rb")
test = pickle.load(file)
file = open('Vocabulary.pkl', "rb")
voc = pickle.load(file)

file = open('info/annotation_training.pkl', "rb")
atrain = pickle.load(file, encoding='latin1')

file = open('info/annotation_validation.pkl', "rb")
aval = pickle.load(file, encoding='latin1')

file = open('info/annotation_test.pkl', "rb")
atest = pickle.load(file, encoding='latin1')


### Features & Labels ###
Y_A_TR,Y_C_TR,Y_E_TR,Y_I_TR,Y_N_TR,Y_O_TR = [],[],[],[],[],[]
Y_A_VL,Y_C_VL,Y_E_VL,Y_I_VL,Y_N_VL,Y_O_VL = [],[],[],[],[],[]
Y_A_TS,Y_C_TS,Y_E_TS,Y_I_TS,Y_N_TS,Y_O_TS = [],[],[],[],[],[]

X_TR = np.zeros([6000,(300*longest())])
X_VL = np.zeros([2000,(300*longest())])
X_TS = np.zeros([2000,(300*longest())])

i = 0
for key in train:
    j = 0
    Y_A_TR.append(atrain['agreeableness'][key])
    Y_C_TR.append(atrain['conscientiousness'][key])
    Y_E_TR.append(atrain['extraversion'][key])
    Y_I_TR.append(atrain['interview'][key])
    Y_N_TR.append(atrain['neuroticism'][key])
    Y_O_TR.append(atrain['openness'][key])
    for token in train[key]:
        if token in voc.keys():
           X_TR[i][j:j+300] = voc[token]
        j = j + 300
    i = i + 1
    
i = 0
for key in val:
    j = 0
    Y_A_VL.append(aval['agreeableness'][key])
    Y_C_VL.append(aval['conscientiousness'][key])
    Y_E_VL.append(aval['extraversion'][key])
    Y_I_VL.append(aval['interview'][key])
    Y_N_VL.append(aval['neuroticism'][key])
    Y_O_VL.append(aval['openness'][key])
    for token in val[key]:
        if token in voc.keys():
           X_VL[i][j:j+300] = voc[token]
        j = j + 300
    i = i + 1
    
    
i = 0
for key in test:
    j = 0
    Y_A_TS.append(atest['agreeableness'][key])
    Y_C_TS.append(atest['conscientiousness'][key])
    Y_E_TS.append(atest['extraversion'][key])
    Y_I_TS.append(atest['interview'][key])
    Y_N_TS.append(atest['neuroticism'][key])
    Y_O_TS.append(atest['openness'][key])
    for token in test[key]:
        if token in voc.keys():
           X_TS[i][j:j+300] = voc[token]
        j = j + 300
    i = i + 1
    
MODEL = SVR(kernel= 'rbf', C= 1e3, gamma= 0.1)
MODEL.fit(X_TR,Y_I_TR)
YP_I_VL = MODEL.predict(X_VL)
YP_I_TS = MODEL.predict(X_TS)
mae_vl = 1 - mean_absolute_error(Y_I_VL, YP_I_VL)
mae_ts = 1 - mean_absolute_error(Y_I_TS, YP_I_TS)
print('MAE VAL = ' + str(mae_vl))
print('MAE TS = ' + str(mae_ts))

