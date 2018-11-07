import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.decomposition import TruncatedSVD
import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import warnings
warnings.filterwarnings('ignore')
from src.models.plt import plot_cluster


N_CLUSTERS = 2
WHRITE_CLUSTER = True
WHRITE_PLOT = True

if __name__=='__main__':
    
    data = pd.read_csv('features/text/tfidf_doc.csv',sep='§')

    model = KMeans(n_clusters=2,random_state=42,n_init=30)

    cluster = model.fit_predict(data.drop(['Sequence'],axis = 'columns'))

    data['cluster'] = cluster

    cluster = pd.Series(cluster)
    cluster.name = 'Cluster'

    if WHRITE_CLUSTER:
        result = pd.concat([data['Sequence'],cluster],axis=1)
        result.to_csv('result/cluster_tfidf.csv',sep='§')

    svd = TruncatedSVD(n_components=2)
    svd = pd.DataFrame(svd.fit_transform(data.drop(['Sequence','cluster'],axis='columns')))
    svd = svd.add_prefix(f'Svd_')

    plt.scatter(svd['Svd_0'], svd['Svd_1'], c=cluster, s=50, cmap='viridis')
    plt.show()


    if WHRITE_PLOT:
        plot_cluster(svd[['Svd_0','Svd_1']].values,data['Sequence'],cluster.values,'result/plot_cluster_tfidf.html')

