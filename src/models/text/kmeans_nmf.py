import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.decomposition import TruncatedSVD
import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()  # for plot styling
import warnings
warnings.filterwarnings('ignore')
from plt import plot_cluster
from evaluate_cluster import best_k

N_COMPONENTS = 2
N_CLUSTERS = 2
WHRITE_CLUSTER = False
WHRITE_PLOT = False

if __name__ == '__main__':

    data = pd.read_csv(f'features/text/nmf_tfidf_{N_COMPONENTS}.csv', sep='§')

    model = KMeans(n_clusters=best_k(data.drop(
        ['Sequence'], axis='columns'), verbose=False), random_state=42, n_init=30)

    cluster = model.fit_predict(data.drop(['Sequence'], axis='columns'))

    data['cluster'] = cluster

    # plt.scatter(data['nmf_0'], data['nmf_1'], c=cluster, s=50, cmap='viridis')
    # plt.show()

    if WHRITE_CLUSTER:
        result = data[['Sequence', 'cluster']]
        result.to_csv('result/cluster_tfidf.csv', sep='§')

    if WHRITE_PLOT:
        plot_cluster(data[['nmf_0', 'nmf_1']].values, data['Sequence'],
                     cluster,
                     f'result/plot_cluster_{best_k(data.drop(['Sequence'],axis = 'columns'), verbose=False)}_nmf.html')
