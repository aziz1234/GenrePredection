# -*- coding: utf-8 -*-
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
from preprocess import final_df
from preprocess import clean_sentence

warnings.filterwarnings("ignore")
nltk.download("stopwords")
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

data = final_df()


genres = data.genre.value_counts().reset_index()['index']

vectorizer = {'tfidf':TfidfVectorizer()}

classifier = {
              'multinomial_nb': MultinomialNB()}

n_vec = len(vectorizer)
n_clf = len(classifier)

n_iterations = n_vec*n_clf

temp = data.copy()

test_result = {}

for genre in genres[0:7]:
    temp['genre_y'] = [1 if y == genre else 0 for y in temp['genre']]
    k = 1
    final_score = {}
    for vect_name, vect in vectorizer.items():
        for clf_name, clf in classifier.items():
            print('>> ' + genre.upper() + ' [' + str(k) + '/' +str(n_iterations) + ']: ' + vect_name + ' - ' + clf_name )
            kf = KFold(n_splits=3, random_state=None, shuffle=False)
            
            acc_normal = []
            prec_normal = []
            rec_normal = []
            f1_normal = []
            auc_normal = []
            
            acc_over = []
            prec_over = []
            rec_over = []
            f1_over = []
            auc_over = []
            
            acc_under = []
            prec_under = []
            rec_under = []
            f1_under = []
            auc_under = []
            
            i = 1
            x, y = temp.plot_clean, temp.genre_y
            for train_index, test_index in kf.split(data):
                x_train, y_train = x.iloc[train_index], y.iloc[train_index]
                x_test, y_test = x.iloc[test_index], y.iloc[test_index]
                
                train_vect = vect.fit_transform(x_train)
                
                clf.fit(train_vect, y_train)
                
                test_vect = vect.transform(x_test)
                y_pred = clf.predict(test_vect)

                acc_normal.append(np.mean(y_pred==y_test))
                prec_normal.append(precision_score(y_test, y_pred))
                rec_normal.append(recall_score(y_test, y_pred))
                f1_normal.append(f1_score(y_test, y_pred))
                auc_normal.append(roc_auc_score(y_test, y_pred))
#                 print('NORMAL: iteration ' + str(i) + ': ' + str(get_raw_from_cm(confusion_matrix(y_test, y_pred))))
                
                train_vect_over, y_train_over = SMOTE().fit_sample(train_vect, y_train) 
                clf.fit(train_vect_over, y_train_over)
                
                test_vect = vect.transform(x_test)
                y_pred = clf.predict(test_vect)

                acc_over.append(np.mean(y_pred==y_test))
                prec_over.append(precision_score(y_test, y_pred))
                rec_over.append(recall_score(y_test, y_pred))
                f1_over.append(f1_score(y_test, y_pred))
                auc_over.append(roc_auc_score(y_test, y_pred))
#                 print('Overfitting: iteration ' + str(i) + ': ' + str(get_raw_from_cm(confusion_matrix(y_test, y_pred))))
                
                train_vect_under, y_train_under = EditedNearestNeighbours().fit_sample(train_vect, y_train) 
                clf.fit(train_vect_under, y_train_under)
                
                test_vect = vect.transform(x_test)
                y_pred = clf.predict(test_vect)

                acc_under.append(np.mean(y_pred==y_test))
                prec_under.append(precision_score(y_test, y_pred))
                rec_under.append(recall_score(y_test, y_pred))
                f1_under.append(f1_score(y_test, y_pred))
                auc_under.append(roc_auc_score(y_test, y_pred))
#                 print('Underfitting: iteration ' + str(i) + ': ' + str(get_raw_from_cm(confusion_matrix(y_test, y_pred))))
#                 print()
                
                
                i+=1
                
            print('Summary Normal: ' + 
                  'accuracy = ' + str(round(np.mean(acc_normal),2)) + ' | ' + 
                  'precision = ' + str(round(np.mean(prec_normal),2)) + ' | ' +
                  'recall = ' + str(round(np.mean(rec_normal),2)) + ' | ' +
                  'f1 = ' + str(round(np.mean(f1_normal),2)) + ' | ' + 
                  'auc = ' + str(round(np.mean(auc_normal),2))
                 )
            
            print('Summary Over Sampling: ' + 
                  'accuracy = ' + str(round(np.mean(acc_over),2)) + ' | ' + 
                  'precision = ' + str(round(np.mean(prec_over),2)) + ' | ' +
                  'recall = ' + str(round(np.mean(rec_over),2)) + ' | ' +
                  'f1 = ' + str(round(np.mean(f1_over),2)) + ' | ' + 
                  'auc = ' + str(round(np.mean(auc_over),2))
                 )
            
            print('Summary Under Sampling: ' + 
                  'accuracy = ' + str(round(np.mean(acc_under),2)) + ' | ' + 
                  'precision = ' + str(round(np.mean(prec_under),2)) + ' | ' +
                  'recall = ' + str(round(np.mean(rec_under),2)) + ' | ' +
                  'f1 = ' + str(round(np.mean(f1_under),2)) + ' | ' + 
                  'auc = ' + str(round(np.mean(auc_under),2))
                 )
            print()
            k+=1
            final_score[(vect_name, clf_name, 'over')] = np.mean(auc_over)
            final_score[(vect_name, clf_name, 'normal')] = np.mean(auc_normal)
            final_score[(vect_name, clf_name, 'under')] = np.mean(auc_under)
        print('------')
    test_result[genre] = max(final_score, key=final_score.get)
    
def predict_genre(s, pipe_dict):
    s_new = clean_sentence(s)
    genre_analyzed = []
    proba = []
    for genre, pipe in pipe_dict.items():
        res = pipe.predict_proba([s_new])
        genre_analyzed.append(genre)
        proba.append(res[0][1])
    data = pd.DataFrame({'genre': genre_analyzed, 'proba': proba})
    data = data.sort_values(by='proba', ascending=True)
    ax = data.plot(x='genre', y='proba', kind='barh')
    plt.show()
    

#def get_test_result():
#    return test_result
#def get_data():
#    return data
def get_dict():
    pipe_dict = {}
    for genre in genres[:7]:
        data['genre_y'] = [1 if y == genre else 0 for y in data['genre']]
        vect_name = test_result[genre][0]
        clf_name = test_result[genre][1]
        sampling_name = test_result[genre][2]

        vect = clone(vectorizer[vect_name])
        clf = clone(classifier[clf_name])
    
        x_vect = vect.fit_transform(data.plot_clean)
    
        if sampling_name == 'normal':
            clf.fit(x_vect, data.genre_y)
        elif sampling_name == 'over':
            x_vect, y = SMOTE().fit_sample(x_vect, data.genre_y)
            clf.fit(x_vect, y)
        elif sampling_name == 'under':
            x_vect, y = EditedNearestNeighbours().fit_sample(x_vect, data.genre_y)
            clf.fit(x_vect, y)
        pipe_dict[genre] = Pipeline([('vect', vect), ('clf', clf)])
    return pipe_dict