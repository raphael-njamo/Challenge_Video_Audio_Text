import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, plot, iplot


def plot_cluster(coords, names, labels, name_plot):
    ''' 
    Create a scatter plot
    
    Arguments:
        coords {numpy array} -- A numpy array with N lines and 2 columns 
                                (N=number of individuals) each column 
                                correspond to a dimension
        names {list} -- list corresponding to the names of the individuals
        labels {type} -- Label of the cluster (an integer like 0 for the first cluster, 1 for the second...)
        name_plot {str} -- name of the html file of the plot
    '''

    # Create a trace
    trace = go.Scatter(
        x = coords[:, 0],
        y = coords[:, 1],
        mode = 'markers',
        text = names,
        marker = dict(
            size = 10,
            color = labels,
            line = dict(
                width = 2,
                color = 'rgb(0, 0, 0)'
            )
        )
    )

    data = [trace]

    layout = dict(title = 'Styled Scatter',
                    yaxis = dict(zeroline = False),
                    xaxis = dict(zeroline = False)
                    )

    fig = dict(data=data, layout=layout)
    plot(fig, filename=name_plot)


if __name__ == '__main__':

    data = pd.read_csv('features/merged/merged.csv', sep='§')
    del data['Unnamed: 0']
    model = KMeans(n_clusters=3, random_state=42, n_init=30)
    cluster = model.fit_predict(normalize(data.drop(['Sequence'], axis = 'columns')))
    data['cluster']= cluster

    tsne = TSNE(n_components=2, random_state=42)
    tsne = tsne.fit_transform(normalize(data.drop(['Sequence','cluster'], axis = 'columns')))
    plot_cluster(tsne,data['Sequence'],
            cluster,
            f'result/plot_cluster_{3}_merge.html')
