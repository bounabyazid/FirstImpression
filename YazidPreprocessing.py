import re
import nltk
import math
import scipy

import gensim
import pickle as pkl

from math import log
from scipy.stats import logistic
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic as wn_ic

from nltk.corpus import sentiwordnet as wsn

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from nltk.tokenize import RegexpTokenizer

from RegExpReplacement import RegexpReplacer
from RemoveNegatifAntonim import AntonymReplacer

from string import punctuation
from nltk import word_tokenize
from collections import defaultdict

from sklearn.feature_extraction.text import CountVectorizer

from matplotlib import pyplot as plt

from gensim.models import Word2Vec

import inflect
from urlextract import URLExtract
#____________________________________________________________________

def split_uppercase(value):
    S = re.sub(r'([A-Z][a-z]+)', r' \1', value)
    return re.sub(r'([A-Z]+)', r' \1', S)
#____________________________________________________________________

def Convert_Numbers_2_Words(number):
    p = inflect.engine()
    return p.number_to_words(number)
#____________________________________________________________________

def Extract_Numbers(string, ints=True):            
    numexp = re.compile(r'\d[\d,]*[\.]?[\d{2}]* ?')
    numbers = numexp.findall(string)
    numbers = [x.strip(' ') for x in numbers]

    return numbers
#____________________________________________________________________
def Extract_URLs(sentence):
    extractor = URLExtract()
    urls = extractor.find_urls(sentence)
    return urls
#____________________________________________________________________

def Extract_emails(sentence):
    regex = re.compile(("([a-z0-9!#$%&'*+\/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+\/=?^_`"
                    "{|}~-]+)*(@|\sat\s)(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?(\.|"
                    "\sdot\s))+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)"))

    emails = re.findall(r'[\w\.-]+@[\w\.-]+', sentence)
    return emails
#____________________________________________________________________

def Extract_Acronyms(sentence):
    Acronyms = re.findall('(?:(?<=\.|\s)[A-Z]\.)+',sentence)
    return Acronyms
#____________________________________________________________________

def List_Punctuations(text):
    text = re.sub('[A-Za-z]|[0-9]|[\n\t]|[£€$%& ]','',text)
    SymbList = list(dict.fromkeys(text).keys())
    #print(SymbList)
    return SymbList
#____________________________________________________________________

def Remove_From_Sentence(sentence,List):
    for ele in List:
        sentence = sentence.replace(str(ele),'')
    return sentence
#____________________________________________________________________

def Remove_From_Sentence(subString):
    
    return True
#____________________________________________________________________

def give_emoji_free_text(text):
    allchars = [str for str in text]
    emoji_list = [c for c in allchars if c in emoji.UNICODE_EMOJI]
    clean_text = ' '.join([str for str in text.split() if not any(i in str for i in emoji_list)])
    return clean_text

#____________________________________________________________________

def Remove_punctuations(text):
    SymbList =  [':', '.', '¿', '?', ',', '…', ';', '/', '-', '¡', '!', '|', 'ブ', 'レ', 'イ', 'ジ', 'ン', 'グ', 'ス', 'タ', 'ー', '+', '’', '–', '=', '“', '”', '‘', '„', '*', '_', '´']
    return ''.join(c for c in text if c not in SymbList)
#____________________________________________________________________

def URL_Removal(sentence):
    urls = re.findall('(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+.com', sentence)
    for url in urls:
        sentence = sentence.replace(url,'')

    urls1 = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', sentence)

    for url in urls1:
        sentence = sentence.replace(url,'')

    urls2 = re.findall('www.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', sentence)

    for url in urls2:
        sentence = sentence.replace(url,'')
    return sentence
#____________________________________________________________________

def Replace_Punctuations(sentence):
    sentence = sentence.replace('.',' ')
    sentence = sentence.replace('\'',' ')
    return sentence
#____________________________________________________________________

def normalise(word):
    lemmatizer = nltk.WordNetLemmatizer()
    stemmer = nltk.stem.porter.PorterStemmer()

    word = word.lower()
    word = stemmer.stem_word(word)
    word = lemmatizer.lemmatize(word)
    return word
#____________________________________________________________________

def normalise_words(words):
    Nwords = []
    for word in words:
        Nwords.append(normalise(word))
    return Nwords
#____________________________________________________________________

def tokenize(text):
    stop_words = list(set(('a','also','an','and','as','at','but','by','for','from','in','it','its','of','on','or','that','the','to','may','is')))#+list(punctuation)
    stop_words.extend(list(set(stopwords.words('english'))))
    text = Replace_Punctuations(text)  
    words = word_tokenize(text)

    #words = normalise_words(words)
    
    return [w for w in words if w not in stop_words and not w.isdigit() and len(w)>1]
#____________________________________________________________________

def PREPROCESSING(sentence):
    S = sentence
    S = S.lower()

    RegReplacer = RegexpReplacer()
    S = RegReplacer.replace(S)

    tokens = tokenize(S)

    return tokens
#____________________________________________________________________

def Word_Antonyms(word):
    antonyms = []
    for syn in wn.synsets(word):
        for l in syn.lemmas():
            if l.antonyms():
               antonyms.append(l.antonyms()[0].name())
    return antonyms
#____________________________________________________________________

def EMBEDDING_SIMILARITY(tokens1,tokens2):

    VOC = LOAD_VOCABULARY()

    n = len(tokens1)
    m = len(tokens2)

    sim1 = 0
    for word1 in tokens1:
        if word1 in tokens2:
           sim1 = sim1 + 1
        else:
            S1 = VOC[word1]
            maxsim = -1
            for word2 in tokens2:
                S2 = VOC[word2]
                cos = cosine_sim(S1,S2)    
                if cos > maxsim:
                   maxsim = cos
            if maxsim != -1:
               sim1 = sim1 + maxsim
    sim1 = round(sim1/n,2)

    print('Sim1(S1,S2) = '+str(sim1))

    sim2 = 0
    for word2 in tokens2:
        if word2 in tokens1:
           sim2 = sim2 + 1
        else:
            S2 = VOC[word2]
            maxsim = -1
            for word1 in tokens1:
                S1 = VOC[word1]
                cos = cosine_sim(S1,S2)    
                if cos > maxsim:
                   maxsim = cos
            if maxsim != -1:
               sim2 = sim2 + maxsim
    sim2 = round(sim2/m,2)

    print('Sim2(S2,S1) = '+str(sim2))

    sim = (sim1 + sim2)/2
    #sim = sim1 if m>=n else sim2
    #sim = sim1 if sim2>=sim1 else sim2
    
    return round(sim,2)
#________________________________________________________________________