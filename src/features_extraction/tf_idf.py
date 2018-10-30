import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
from nltk.corpus import stopwords


stop = set(stopwords.words('french'))
stop.update(['.', ',', '"', "'", '?', '!', ':',
                   ';', '(', ')', '[', ']', '{', '}','-'])

data = pd.read_csv('features/text/sequence_text.csv', sep='§')

data['Text'] = data['Text'].apply(
    lambda x: ' '.join([i for i in word_tokenize(x) if i not in stop]))

tfidf = TfidfVectorizer(max_df=0.95, min_df=2)
tfidf = pd.DataFrame(tfidf.fit_transform(data['Text']).todense())

data = pd.concat([data,tfidf],axis = 'rows')
data.drop(['Text'],axis='columns')

data.to_csv('features/text/tfidf.csv',index=False,sep='§')
