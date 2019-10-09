import re

import pickle
import gensim
import numpy, scipy.io
import YazidPreprocessing

RegReplacer = YazidPreprocessing.RegexpReplacer()

def WritePickle(file,Data):
    with open(file+'.pkl', 'wb') as handle:
        pickle.dump(Data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def ReadPickle(file):
    with open(file+'.pkl', 'rb') as handle:
        Data = pickle.load(handle)
    return Data

def PreprocessTranscript(Transcript,Urls,Emails,Punctuations):
    Text = re.sub('\[.*?\]', '', Transcript)
    Text = RegReplacer.replace(Text)

    for Url in Urls:
        if Url in Text:
           Text = Text.replace(Url,' link ')    
    for Email in Emails:
        if Email in Text:
           Text = Text.replace(Email,' Email ') 
    for Punctuation in Punctuations:
        Text = Text.replace(Punctuation,' ')
        
    Text = re.sub(r"\s+", " ",Text)
    Text = Text.strip()
        
    #remember replacing numbers  just in case
    
    #return Text
    return YazidPreprocessing.tokenize(Text)
    
def Preprocessing_Taranscriptions(transcriptions,Urls,Emails,Punctuations):
    Clean_trans = {}
    for key in transcriptions.keys():
        #transcriptions[key] = PreprocessTranscript(transcriptions[key],Urls,Emails,Punctuations)
        Clean_trans[key]=PreprocessTranscript(transcriptions[key],Urls,Emails,Punctuations)
   
    return Clean_trans
    return transcriptions

def Preprocessing_Dataset():
    file = open('info/transcription_training.pkl', "rb")
    tran_train = pickle.load(file, encoding='latin1')
    Clean_trans_train = []

    file = open('info/transcription_validation.pkl', "rb")
    tran_val = pickle.load(file, encoding='latin1')
    Clean_tran_val = []
    
    file = open('info/transcription_test.pkl', "rb")
    tran_test = pickle.load(file, encoding='latin1')
    Clean_tran_test = []
    
    Text = ' '.join(tran_train.values())+' '.join(tran_val.values())+' '.join(tran_test.values())
    
    Punctuations = YazidPreprocessing.List_Punctuations(Text)
    Urls = YazidPreprocessing.Extract_URLs(Text)
    Emails = YazidPreprocessing.Extract_emails(Text)
    
    print(Punctuations)
    print('________________________')
    print(Urls)
    print('________________________')
    print(Emails)
    print('________________________')
    
#    print(tran_test['-10-QQDO_ME.001.mp4'])
#    print('__________________________________')
#    print(PreprocessTranscript(tran_test['-10-QQDO_ME.001.mp4'],Urls,Emails,Punctuations))

    Clean_trans_train = Preprocessing_Taranscriptions(tran_train,Urls,Emails,Punctuations)
    Clean_tran_val = Preprocessing_Taranscriptions(tran_val,Urls,Emails,Punctuations)
    Clean_tran_test = Preprocessing_Taranscriptions(tran_test,Urls,Emails,Punctuations)
    
    return Clean_trans_train,Clean_tran_val,Clean_tran_test

def Feature_Extraction():
    Clean_trans_train,Clean_tran_val,Clean_tran_test = Preprocessing_Dataset()
    
    Words = [item for sublist in Clean_trans_train.values() for item in sublist]
    Words.extend([item for sublist in Clean_tran_val.values() for item in sublist])
    Words.extend([item for sublist in Clean_tran_test.values() for item in sublist])

    Words = list(set(Words))
    
    print('Loading Google Pre-trained model ...')
    model = gensim.models.KeyedVectors.load_word2vec_format('/home/polo/Downloads/GoogleNews-vectors-negative300.bin', binary=True)
    print('Google Pre-trained model has been loaded')

    WV = model.wv
    Vocabulary = {}
    
    for word in Words:
        if word in WV.vocab:
           Vocabulary[word] = WV[word]

    WritePickle('Clean_trans_train',Clean_trans_train)
    WritePickle('Clean_tran_val',Clean_tran_val)
    WritePickle('Clean_tran_test',Clean_tran_test)
    WritePickle('Vocabulary',Vocabulary)
    
    return Clean_trans_train,Clean_tran_val,Clean_tran_test,Vocabulary

Clean_trans_train,Clean_tran_val,Clean_tran_test,Vocabulary = Feature_Extraction()

S = 'Going to be very little  Not that there is none  There are loan programs out there that reward higher than a 720 but for the most part once you get over that  it is just you bragging to your friends that you have a higher credit score than them  or a husband and a wife battling about who has the better credit  Hopefully that an'