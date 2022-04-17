import json

import joblib
import nltk
import pandas as pd
import plotly

nltk.download('averaged_perceptron_tagger')

import os
import sys

from flask import Flask, jsonify, render_template, request
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from plotly.graph_objs import Bar
from sqlalchemy import create_engine

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parentdir)
from models.features import CountWordsBeforeVerb

app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)
# load model
model = joblib.load("models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    # graph 1
    title1 = 'Distribution of Message Genres'
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # graph 2
    title2 = 'Distribution of Message Category'
    col_list = [col for col in df.columns if col.endswith('_related')]
    class_counts = df[col_list].sum().sort_values(ascending=True)
    class_names = list(class_counts.index)
    
    # graph 3
    title3 = 'Top Message Content'
    n = 12
    category_counts = df.drop(columns=['id', 'message', 'original', 'genre', 'request', 'offer', 'related', 
                                       'aid_related', 'weather_related', 'infrastructure_related',                                                                                                              'direct_report']).sum().sort_values(ascending=False)
    rest = category_counts.iloc[n:]
    category_counts = category_counts.iloc[:n]
    category_counts['other'] = rest.sum()
    category_names = list(category_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [

                {

                'data': [

                    Bar(

                        x=genre_names,
                        y=genre_counts

                        )

                        ],

                'layout': {

                    'title': title1,
                    'xanchor': 'center',
                    'width': 1000,
                    'height': 500,


                    'yaxis': {

                        'title': "Count"

                            },

                    'xaxis': {

                        'title': "Genre"

                            }

                            }

                    },

                    {

                    'data': [

                        Bar(

                            x=class_counts,
                            y=class_names,
                            orientation='h'

                            )

                            ],

                    'layout': {

                        'title': title2,
                        "width": 1000,
                        "height": 500,

                        'yaxis': {

                        'automargin': True,

                                },

                        'xaxis': {

                        'title': "Count",
                        'automargin': True,

                                  }

                            }

                },
                {
                    
                 'data': [
                         {
                          'uid': 'f4de1f',
                          'hole': 0.1,
                        #  'name': 'Col2',
                          'pull': 0,
                          'type': 'pie',
                          'domain': {
                                    'x': [
                                            0,
                                            1
                                         ],
                                    'y': [
                                            0,
                                            1
                                         ]
                                     },
                        'marker': {
                                   'colors': [
                                              '#7fc97f',
                                              '#beaed4',
                                              '#fdc086',
                                              '#ffff99',
                                              '#386cb0',
                                              '#f0027f'
                                            ]
                                   },
                      'textinfo': 'label+value',
                      'hoverinfo': 'all',
                      'labels': category_names,
                      'values': list(category_counts),
                      'showlegend': False
                    }
                    ],
            'layout': {
                        'title': title3,
                        'width': 800,
                        'height': 570,
                        'autosize': False,
                        'xaxis': { 'xanchor':
                                        "center"
                                 }   
                      }
                }
            ]
    
    # encode plotly graphs in JSON
    ids = ['graph-{}'.format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
