import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import re



def make_lower_case(text):
    return text.lower()

def compute_length(text):
    return len(text.split(' '))

def remove_stop_words(text):
    text = text.split()
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]
    texts = [w for w in text if w.isalpha()]
    texts = " ".join(texts)
    return texts

# Function for removing punctuation
def remove_punctuation(text):
    tokenizer = RegexpTokenizer(r'\w+')
    text = tokenizer.tokenize(text)
    text = " ".join(text)
    return text

# Function for removing the html tags
def remove_html(text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)


if __name__ == "__main__":
    data1 = pd.read_csv('allCombined-Aryaman.csv')
    data2 = pd.read_csv('allCombined-Divyansh.csv')

    data = pd.concat([data1, data2], sort=False)
    data = data.dropna()
    data = data.drop_duplicates(subset=None, keep='first', inplace=False)


    data['cleaned_desc'] = data['text'].apply(func = make_lower_case)
    data['cleaned_desc'] = data.cleaned_desc.apply(func = remove_stop_words)
    data['cleaned_desc'] = data.cleaned_desc.apply(func=remove_punctuation)
    data['cleaned_desc'] = data.cleaned_desc.apply(func=remove_html)
    data = data.drop_duplicates(subset='cleaned_desc', keep='first', inplace=False)
    data.reset_index(level=0, inplace=True)
    data.rename(columns = {'index':'ID'}, inplace = True)

    tf = TfidfVectorizer(analyzer='word',stop_words='english',max_df=0.8,min_df=0.1,use_idf=True,ngram_range=(1,3))
    tfidf_matrix = tf.fit_transform(data['cleaned_desc'])

    model_tf_idf = NearestNeighbors(algorithm='brute', metric='cosine')
    model_tf_idf.fit(tfidf_matrix)

    model = pickle.dump(model_tf_idf, open('model.pkl', 'wb'))
    data.to_csv('articles.csv')