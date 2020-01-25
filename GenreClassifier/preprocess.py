import pandas as pd 
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import wordnet as wn
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from nltk.corpus import stopwords
from sklearn.base import clone
from bs4 import BeautifulSoup
import re

warnings.filterwarnings("ignore")
nltk.download("stopwords")
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

def decode_str_dict(x):
    try:
        out=eval(x)[0]['name']
    except:
        try:
            out=eval(x)['name']
        except:
            try:
                out=eval(x)
            except:
                out=x
    return out



#data['genre']=data['genre'].apply(decode_str_dict)


def clean_sentence(sentence):
    

    sentence = sentence.lower()
    wordnet_lemmatizer = WordNetLemmatizer()
    stop = set(stopwords.words('english'))
    VERB_CODES = {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', }
    
    new_sentence = []
    words = nltk.word_tokenize(sentence)
    role = nltk.pos_tag(words)
    for i, word in enumerate(words):
        if role[i][1] in VERB_CODES: 
            new_word = wordnet_lemmatizer.lemmatize(word, 'v')
        else:
            new_word = wordnet_lemmatizer.lemmatize(word)
        if new_word not in stop and new_word.isalpha():
            new_sentence.append(new_word)
        
    s = ' '.join(new_sentence)
    s = s.replace("n't", " not")
    s = s.replace("'s", " is")
    s = s.replace("'re"," are")
    s = s.replace("'d", " would")
    s = s.replace("'ve", " have")
    s = s.replace("'ll", " will")
    s = s.replace("'m", " am")
    return s


#genres = data.genre.value_counts().reset_index()['index']
def final_df():
    data = pd.read_csv('D:/GenreClassifier/movies_metadata2.csv' ,encoding='latin-1',low_memory=False, dtype = str)
    #data= data.drop(['adult', 'belongs_to_collection', 'budget', 'homepage',	'imdb_id',	'original_language',	'popularity',	'poster_path',	'production_companies',	'production_countries', 'release_date', 'revenue',	'runtime',	'spoken_languages',	'status', 'tagline', 'title', 'video', 'vote_average', 'vote_count', 'id'], axis='columns')
    data.head()
    data = data.drop(['imdbID' ,'Genre2' ,'Genre3'],axis=1)
    #data.columns = ['genre', 'title', 'plot'] 
    data.columns = ['title', 'plot','genre']
    data=data.mask(data.applymap(str).eq('[]'))
    print("done1")
    #data['genre']=data['genre'].apply(decode_str_dict)
    #print("done2")
    data['plot_clean'] = data['plot'].astype(str).apply(clean_sentence)
    print("done3")
    return data

