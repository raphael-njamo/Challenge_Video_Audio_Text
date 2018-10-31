import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation


N_COMPONENTS = 2
TF_IDF = False

if TF_IDF:
    data = pd.read_csv('features/text/tfidf_doc.csv',sep='§')
else :
    data = pd.read_csv('features/text/tf_doc.csv',sep='§')
    


lda = LatentDirichletAllocation(n_components=N_COMPONENTS, max_iter=5,random_state=42)
lda = pd.DataFrame(lda.fit_transform(data.drop(['Sequence'],axis='columns')))
lda = lda.add_prefix(f'LDA_')
lda = pd.concat([data['Sequence'],lda],axis='columns')
if TF_IDF:
    lda.to_csv(f'features/text/lda_tfidf_{N_COMPONENTS}.csv',index=False,sep='§')
else: 
    lda.to_csv(f'features/text/lda_tf_{N_COMPONENTS}.csv',index=False,sep='§')