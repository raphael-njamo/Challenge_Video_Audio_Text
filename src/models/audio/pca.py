# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 17:20:46 2018

@author: linan
"""
import numpy as np
from sklearn.decomposition import PCA
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, plot, iplot
import pandas as pd

import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, plot, iplot
import numpy as np

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
    pca = PCA(n_components=2)
    
    pca.fit(df)
    trans = pca.transform(df)
    
    print(type(trans))
    
    plt.scatter(trans[:,0], trans[:,1],c=label)
       
    label = pd.Series(label)
    labels_0 = label == 0
    
    plot_cluster(trans, df.index, label, 'test.html')
