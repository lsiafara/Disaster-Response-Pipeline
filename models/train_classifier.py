import pickle
import re
import sys

import nltk
import numpy as np
import pandas as pd
import sqlalchemy

nltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger', 'omw-1.4'])

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import (CountVectorizer, TfidfTransformer,
                                             TfidfVectorizer)
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.tree import DecisionTreeClassifier

from models.features import CountWordsBeforeVerb


def load_data(database_filepath):
    '''
    INPUT 
        database_filepath - the path to the database file
        
    OUTPUT
        X - an array of strings holding the disaster messages
        Y - an array of 36 columns (categories) holding the label flag (0/1)
        category_names -  a list with the names of the categories
    '''
        
    engine = sqlalchemy.create_engine('sqlite:///' + database_filepath)
    
    df = pd.read_sql_table('DisasterResponse', engine)
    df = df[df['related']!=2]
    df.dropna(axis=0, inplace=True)
    
    X = df['message'].str.lower().values
    
    Y_df = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    Y = Y_df.values
    
    category_names = list(Y_df.columns)
    
    return X, Y, category_names


def tokenize(text):
    '''
    INPUT 
        text - a string to tokenize
        
    OUTPUT
        words - a list of words resulting from the split text
    '''
    
    words = word_tokenize(text.lower())
    words = [w for w in words if w not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(word) for word in words]
    return lemmas


def build_model():
    '''
    This function returns a multioutput classifier with optimized hyperparameters
    OUTPUT
        model - the classifier with optimized hyperparameters for message classification
    '''
    
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('count_words_before_verb', CountWordsBeforeVerb())
        ])),

        ('clf', RandomForestClassifier())
    ])
    
    parameters = { 'clf__n_estimators': [10, 100],
                  'clf__criterion': ['gini', 'entropy'],
                  'features__text_pipeline__tfidf__norm': ['l2', 'l1'],
              }

    model = GridSearchCV(pipeline, param_grid=parameters, scoring='f1_micro', verbose=3)
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Prints out the classification report results

    INPUT 
        model - a classification model
        X - an array with the test data for the predictions
        Y - an array with the ground truth labels
    '''
    
    Y_predict = model.predict(X_test)
    results = []

    for index,value in enumerate(category_names):
        result = classification_report(Y_test[:,index], Y_predict[:,index])
        results.append(result)
    
        print("{}:\n".format(value))
        print(result)


def save_model(model, model_filepath):
    '''
    Saves the classifier model

    INPUT 
        model - a classification model
        model_filepath - the path to save the model file
    '''
    
    pickle.dump(model, open(model_filepath, 'wb'))

 
def main():
    '''
    Saves the classifier model

    INPUT 
        model - a classification model
        model_filepath - the path to save the model file
    '''

    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
