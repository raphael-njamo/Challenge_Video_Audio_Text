import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

# audio_stat_desc = pd.read_csv('features/audio/Statistique_desc.csv',sep='§')
# audio_stat_desc['Sequence'] =  audio_stat_desc['Sequence'].apply(lambda x: x[:-len('_AUDIO')])
# assert len(audio_stat_desc) == 308
# audio_stat_desc.sort_values('Sequence',inplace=True)
# print(audio_stat_desc.head())


decoupage_seq_son = pd.read_csv('features/audio/Decoupage_Sequence_son.csv', sep='§')
decoupage_seq_son['Sequence'] =  decoupage_seq_son['Sequence'].apply(lambda x: x[:-len('_AUDIO')])
assert len(decoupage_seq_son) == 308
decoupage_seq_son.sort_values('Sequence',inplace=True)
print(decoupage_seq_son.head())

segmentation_parole = pd.read_csv('features/audio/Segmentation_parole.csv', sep='§')
segmentation_parole['Sequence'] =  segmentation_parole['Sequence'].apply(lambda x: x[:-len('_AUDIO')])
assert len(segmentation_parole) == 308
segmentation_parole.sort_values('Sequence',inplace=True)
print(segmentation_parole.head())

taux_parole = pd.read_csv('features/audio/Taux_Parole.csv', sep='§')
assert len(taux_parole) == 308
taux_parole.sort_values('Sequence',inplace=True)

print(taux_parole.head())


cuts = pd.read_csv('features/video/df_cuts.csv', sep='§')
cuts.rename(columns={'Unnamed: 0' : 'Sequence'}, inplace=True)
cuts['Sequence'] =  cuts['Sequence'].apply(lambda x: x[:-len('_VIDEO.mp4')])
assert len(cuts) == 308
cuts.sort_values('Sequence',inplace=True)
print(cuts.head())


histo = pd.read_csv('features/video/df_histo.csv', sep='§')
histo.rename(columns={'Unnamed: 0' : 'Sequence'}, inplace=True)
assert len(histo) == 308
histo.sort_values('Sequence',inplace=True)
histo['Sequence'] =  histo['Sequence'].apply(lambda x: x[:-len('_VIDEO')])
print(histo.head())
pca_histo = PCA(n_components=2)
pca_histo = pd.DataFrame(pca_histo.fit_transform(normalize(histo.drop('Sequence',axis=1))))
pca_histo = pca_histo.add_prefix(f'Histo_')
pca_histo = pd.concat([histo['Sequence'],pca_histo],axis=1)
del histo

print(pca_histo.head())


njamo = pd.read_csv('features/video/featuresVideo.csv',sep='§')
njamo.rename(columns={'Unnamed: 0' : 'Sequence'}, inplace=True)
assert len(njamo) == 308
njamo.sort_values('Sequence',inplace=True)
print(njamo.head())
pca_njamo = PCA(n_components=12)
pca_njamo = pd.DataFrame(pca_njamo.fit_transform(normalize(njamo.drop('Sequence',axis=1))))
pca_njamo = pca_njamo.add_prefix(f'PCA_')
pca_njamo = pd.concat([njamo['Sequence'],pca_njamo],axis=1)
del njamo

print(pca_njamo.head())



emotion = pd.read_csv('features/text/emotion_doc.csv', sep='§')
assert len(emotion) == 308
emotion.sort_values('Sequence',inplace=True)
print(emotion.head())


nmf = pd.read_csv('features/text/nmf_tfidf_5_punct.csv', sep='§')
assert len(nmf) == 308
nmf.sort_values('Sequence',inplace=True)
print(nmf.head())

list_data = [taux_parole,cuts,pca_histo,pca_njamo,emotion,nmf]

data = decoupage_seq_son

for var in list_data:

    data = pd.merge(
            left=data,
            right=var,
            how='left',
            on='Sequence'
        )
del list_data,decoupage_seq_son,taux_parole,cuts,pca_histo,pca_njamo,emotion,nmf


data.to_csv('features/merged/merged.csv', sep='§')
