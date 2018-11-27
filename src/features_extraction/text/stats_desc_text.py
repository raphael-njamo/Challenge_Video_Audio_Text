import pandas as pd
from collections import Counter
from scipy.misc import imread
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import FrenchStemmer
import plotly.graph_objs as go
from sklearn.decomposition import LatentDirichletAllocation
from plotly.offline import download_plotlyjs, plot, iplot
from nltk.corpus import stopwords
import numpy as np 
from wordcloud import WordCloud
import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import warnings


def print_top_words(model, feature_names, n_top_words):
    for index, topic in enumerate(model.components_):
        message = "\nTopic #{}:".format(index)
        message += " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1 :-1]])
        print(message)
        print("="*70)

class LemmaCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(LemmaCountVectorizer, self).build_analyzer()
        return lambda doc: (lemm.stem(w) for w in analyzer(doc))

if __name__ == '__main__':


    data = pd.read_csv('features/text/sequence_text.csv', sep='§')

    data = data.groupby(['Sequence'])['Text'].sum()
    data = data.reset_index()

    print(data.head())

    lemm = FrenchStemmer()

    stop = set(stopwords.words('french'))
    stop.update(['ça','le','la','comme','fait','va','là','cet','quand'])


    tf_vectorizer = LemmaCountVectorizer(max_df=0.95, 
                                     min_df=2,
                                     stop_words=stop,
                                     decode_error='ignore')
    
    text = list(data['Text'].values)
    tf = tf_vectorizer.fit_transform(text)

    feature_names = tf_vectorizer.get_feature_names()
    count_vec = np.asarray(tf.sum(axis=0)).ravel()
    zipped = list(zip(feature_names, count_vec))
    x, y = (list(x) for x in zip(*sorted(zipped, key=lambda x: x[1], reverse=True)))
    # Now I want to extract out on the top 15 and bottom 15 words
    Y = np.concatenate([y[0:15], y[-16:-1]])
    X = np.concatenate([x[0:15], x[-16:-1]])

    # Plotting the Plot.ly plot for the Top 50 word frequencies
    data = [go.Bar(
                x = x[1:51],
                y = y[1:51],
                marker= dict(colorscale='Jet',
                            color = y[0:50]
                            ),
                text='Word counts'
        )]

    layout = go.Layout(
        title='Top 50 Word frequencies after Preprocessing'
    )

    fig = go.Figure(data=data, layout=layout)

    plot(fig, filename='result/most_frequent_word.html')

#######################################################################

    data = pd.read_csv('features/text/sequence_text.csv', sep='§')

    data = data.groupby(['Sequence'])['Text'].sum()
    data = data.reset_index()
    external = pd.read_csv('data/external/corpus_wiki_300.csv', sep='§')
    external['Sequence'] = 'Augmentation'
    data = pd.concat([data,external], axis =0)
    tf_vectorizerl = LemmaCountVectorizer(max_df=0.95, 
                                    min_df=2,
                                    stop_words=stop,
                                    decode_error='ignore')

    textl = list(data['Text'].values)
    tfl = tf_vectorizerl.fit_transform(textl)
    lda = LatentDirichletAllocation(n_components=3, max_iter=10,random_state=42,verbose=1)
    lda.fit(tfl)

    n_top_words = 40
    print("\nTopics in LDA model: ")
    tf_feature_namesl = tf_vectorizerl.get_feature_names()
    print_top_words(lda, tf_feature_namesl, n_top_words)
    
    first_topic = lda.components_[0]
    second_topic = lda.components_[1]
    third_topic = lda.components_[2]
    first_topic_words = [tf_feature_namesl[i] for i in first_topic.argsort()[:-50 - 1 :-1]]
    second_topic_words = [tf_feature_namesl[i] for i in second_topic.argsort()[:-50 - 1 :-1]]
    third_topic_words = [tf_feature_namesl[i] for i in third_topic.argsort()[:-50 - 1 :-1]]

    firstcloud = WordCloud(
                          stopwords=stop,
                          background_color='white',
                          width=2500,
                          height=1800
                         ).generate(" ".join(first_topic_words))
    plt.imshow(firstcloud)
    plt.axis('off')
    plt.show()

    cloud = WordCloud(
                            stopwords=stop,
                            background_color='white',
                            width=2500,
                            height=1800
                            ).generate(" ".join(second_topic_words))
    plt.imshow(cloud)
    plt.axis('off')
    plt.show()

        # Generating the wordcloud with the values under the category dataframe
    cloud = WordCloud(
                            stopwords=stop,
                            background_color='white',
                            width=2500,
                            height=1800
                            ).generate(" ".join(third_topic_words))
    plt.imshow(cloud)
    plt.axis('off')
    plt.show()