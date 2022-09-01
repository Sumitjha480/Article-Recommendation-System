import pandas as pd
import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template
import requests
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import re

app = Flask(__name__)

app.config["DEBUG"] = True

@app.route('/')
def home():
    return render_template('index.html')


@app.errorhandler(404)
def page_not_found(e):
    return "<h1>404</h1><p>Page Not Found</p>", 404



@app.route('/result', methods=['POST'])
def api_filter():
    int_features = [int(x) for x in request.form.values()]
    idx, knx=int_features[0], int_features[1]
    distances, indices = model.kneighbors(tfidf_matrix[idx], n_neighbors=100)
    neighbors = pd.DataFrame({'distance':distances.flatten(), 'id':indices.flatten()})
    nearest_neighbor_cosine = neighbors.merge(articles, left_on='id', right_index=True)

    names=dict()
    for x in range(knx):
        #names.append(nearest_neighbor_cosine['id'][x])
        names[x] = nearest_neighbor_cosine['title'][x]
    return render_template('index.html', prediction_text='Neighbours Are : {}'.format(names))



if __name__ == '__main__':
    articles = pd.read_csv('articles.csv')
    tf = TfidfVectorizer(analyzer='word',stop_words='english',max_df=0.8,min_df=0.1,use_idf=True,ngram_range=(1,3))
    tfidf_matrix = tf.fit_transform(articles['cleaned_desc'].apply(lambda x: np.str_(x)))
    model = pickle.load(open('model.pkl', 'rb'))

    app.run(port=5000, debug=True)