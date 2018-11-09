import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()  # for plot styling
import warnings
from evaluate_cluster import best_k
from plt import plot_cluster


N_CLUSTERS = 3
WHRITE_CLUSTER = True
WHRITE_PLOT = True

if __name__ == '__main__':

    warnings.filterwarnings('ignore')

    data = pd.read_csv('features/text/emotion_doc.csv', sep='§')

    pca = PCA(n_components=2)
    pca = pd.DataFrame(pca.fit_transform(
        data.drop(['Sequence'], axis='columns')))
    pca = pca.add_prefix(f'PCA_')

    model = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=30)

    cluster = model.fit_predict(data.drop(['Sequence'], axis='columns'))

    cluster = pd.Series(cluster)
    cluster.name = 'Cluster'

    if WHRITE_CLUSTER:
        result = pd.concat([data['Sequence'], cluster], axis=1)
        result.to_csv('result/cluster_sentiment.csv', sep='§')

    plt.scatter(pca['PCA_0'], pca['PCA_1'], c=cluster, s=50, cmap='viridis')
    plt.show()

    if WHRITE_PLOT:
        plot_cluster(pca[['PCA_0', 'PCA_1']].values, data['Sequence'],
                     cluster.values,
                     f'result/plot_cluster_{best_k(data.drop(['Sequence'],axis = 'columns'), verbose=False)}_sentiment.html')
