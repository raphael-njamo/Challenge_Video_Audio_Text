import pandas as pd 
from sklearn.decomposition import TruncatedSVD

N_COMPONENTS = 5
data = pd.read_csv('features/text/tfidf_doc_punct.csv',sep='§')

svd = TruncatedSVD(n_components=5)
svd = pd.DataFrame(svd.fit_transform(data.drop(['Sequence'],axis='columns')))
svd = svd.add_prefix(f'Svd_')
svd = pd.concat([data['Sequence'],svd],axis='columns')
svd.to_csv(f'features/text/svd_tfidf_{N_COMPONENTS}_punct.csv',index=False,sep='§')