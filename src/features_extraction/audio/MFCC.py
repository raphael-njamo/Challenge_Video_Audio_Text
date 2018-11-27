# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 14:41:00 2018

@author: linan
"""

import librosa 
import scipy as sp
import numpy as np
import os
from tqdm import tqdm #pour avoir la bar de progression pour les boucles for
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

path = "D:/Challenge_Video_Audio_Text/data/audio/"
infiles = [path + file for file in os.listdir(path) if file.split(".")[-1]=="wav"]

csv = pd.read_csv("D:/Annotations.csv")
csv=csv["Violent"] #♥Exterieur 
csv.drop(csv.index[len(csv)-1], inplace = True)
csv.drop(csv.index[len(csv)-1], inplace = True)

csv.index = range(1, len(csv)+1)

#a utiliser en cas de valeur non numerique dans le csv
#csv = [int(x) if x in ["0", "1"] else 0 for x in csv.values] #list comprehension -> plus rapide que les boucles normal

##--------------------------------------------- TEST
#sr = frequence d'ech nb de valeur par s du signal, ex = 16 000 valeur du signal correspond a 1s 
#y = signal, vecteur de ~ l'intensité/
(sr,y)=sp.io.wavfile.read("D:/Challenge_Video_Audio_Text/data/audio/SEQ_001_AUDIO.wav")
plt.plot(y)

y= y.astype(float)
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
np.shape(mfcc)
#mfcc_delta = librosa.feature.delta(mfcc)

mean_mfcc = np.mean(mfcc, axis=1) #1=ligne et 0=colonne d'apres np (avec pandas c'est l'inverse)
std_mfcc = np.std(mfcc, axis=1)
zcr = librosa.feature.zero_crossing_rate(y) #nb de fois que le signal croise le zero (il change de signe)
np.shape(mean_mfcc)
##-----------------------------------------------------------------------------

#calcul de features
mean_wav = {}
for files in tqdm(infiles) : 
    para = []
    (sr,y)=sp.io.wavfile.read(files)
    
    y= y.astype(float)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    #np.shape(mfcc)
    #mfcc_delta = librosa.feature.delta(mfcc)
    
    #a chaque coeff on a une moyenne sur tous le temps
    mean_mfcc = np.mean(mfcc, axis=1) #1=ligne et 0=colonne d'apres np (avec pandas c'est l'inverse)
    std_mfcc = np.std(mfcc, axis=1)
    zcr = librosa.feature.zero_crossing_rate(y).sum() #nb de fois que le signal croise le zero (il change de signe)
    para.extend(mean_mfcc)
    para.extend(std_mfcc)
    para.append(zcr)
    #np.shape(mean_mfcc)
    mean_wav[files.split('/')[-1]]=para

df = pd.DataFrame.from_dict(mean_wav, orient = "index")

X_train, X_test, y_train, y_test = train_test_split(df, csv, test_size=0.30, random_state=42)

#creation du modele
kmeans = KMeans(n_clusters=2)

#entrainement du modele sur l'ech d'entrainement
kmeans.fit(df) #X_train

#essaie avec l'ech test
label = kmeans.predict(df) #y_pred = ... & X_test

#matrice de confusion
#affiche les resultats d'appartennace au groupe
#y_test = verité terrain et y_pred = prediction
print(confusion_matrix(y_pred, y_test))

#A faire
#acp sur train pour reduire a 2D 
#PCA : http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html










