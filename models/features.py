import pandas as pd
import nltk

from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.base import BaseEstimator, TransformerMixin


class CountWordsBeforeVerb(BaseEstimator, TransformerMixin):

    def count_words_before_verb(self, text):
        '''
        INPUT 
        text - a string with the text
        
        OUTPUT
            X - an array of strings holding the disaster messages
            Y - an array of 36 columns (categories) holding the label flag (0/1)
            category_names -  a list with the names of the categories
        '''
    
        # tokenize by sentences
        sentence_list = nltk.sent_tokenize(text)
        
        counter = 0
        for sentence in sentence_list:
            # tokenize each sentence into words and tag part of speech
            pos_tags = nltk.pos_tag(word_tokenize(sentence.lower()))
            #counter = 0
            for word, tag in pos_tags:
                if tag in ['VB', 'VBD', 'VBN', 'VBG', 'VBZ']:
                    break
                else:
                    counter += 1
        return counter

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        '''
        Applies the count_words_before_verb function to all values in X
        INPUT 
            X - an array of strings 
        
        OUTPUT
            a dataframe with the result of count_words_before_verb function for each row of X 
        '''
        X_tagged = pd.Series(X).apply(self.count_words_before_verb).values

        return pd.DataFrame(X_tagged)