import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from evaluate_cluster import best_k


if __name__ == '__main__':

    data = pd.read_csv('features/text/nmf_tfidf_2.csv', sep='§')

    data = pd.merge(
        left=data,
        right=pd.read_csv('features/audio/Taux_Parole.csv', sep='§'),
        how='left',
        on='Sequence'
    )

    best_k(data.drop(['Sequence'], axis='columns'))
    model = KMeans(n_clusters=4, random_state=42, n_init=30)

    data['Cluster'] = model.fit_predict(data.drop(['Sequence'], axis='columns'))
    print(data.head())
    
    
