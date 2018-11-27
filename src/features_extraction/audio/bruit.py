# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 11:08:12 2018

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



#affiche l'energie de l'enregistrement
def energy(Y, sr, fen=4096, pas=2048):
    """ Fonction qui renvoie l'energie d'un enregistrment audio
    - Y : forme d'onde    
    - sr : taux d'echantillonage
    - On definie une fenetre et un pas """
    
    Y = Y.astype(np.float64)
    N = len(Y)
    eng = []
    for i in range(0,N,pas):
        eng.append(np.log(np.sum(Y[i:i+fen]*Y[i:i+fen])))
    
    return(eng)   




if __name__ == '__main__':
    
    ####################################### Par AUDIO
    sr, y= read("D:/Bruit2.0/SEQ_141_BLOND.wav")
    y = [None for i in y if i == 0]
    y = y[y!=0]
    plt.plot(y)
    
    nrjbruit= energy(y, sr)
    plt.plot(nrjbruit)
    y= y.astype(float)
    zcr = librosa.feature.zero_crossing_rate(y) #frame_length = 2048 fenetre & hop_length = 512 pas
    #somme_zcr = zcr.sum() #nb de fois que le signal croise le zero (il change de signe)
    plt.plot(zcr)
    
    np.mean(zcr)
    np.std(zcr)
    np.min(zcr)
    np.max(zcr)
    
    x= list(zcr)
    plt.plot(x[0])

    #ZCR
#haute frequence -> variation rapide
#basse freq -> variation grande amplitude


    ######################################################## TOUS LES AUDIOS
    path = "D:/Bruit2.0/"
    infiles = [path + file for file in os.listdir(path) if file.split(".")[-1]=="wav"]
    
    zcrbruit = pd.DataFrame()
    for file in infiles:
        sr, y = read(file)
        y = y[y!=0]
        #graph = plt.plot(y)
        nrjbruit= energy(y, sr)
        y= y.astype(float)
        zcr = librosa.feature.zero_crossing_rate(y)
        moy_zcr = np.mean(zcr)
        std_zcr = np.std(zcr)
        min_zcr = np.min(zcr)
        max_zcr = np.max(zcr)
        med_zcr = np.median(zcr)
        #savefig('foo.png')
        zcrbruit = zcrbruit.append({"Sequence":file[12:-10],"Moyenne":moy_zcr ,"Ecart-type": std_zcr ,"Min": min_zcr ,"Max": max_zcr, "Mediane":med_zcr}, ignore_index=True)
    #zcrbruit.to_csv("Feature3_Zcr_bruit.csv", sep="§", index=False)
    
    mo = zcrbruit["Moyenne"]
    np.median(mo)   
    np.max(ma)
    plt.figure(1, figsize=(14, 6))
    plt.hist(mo)
    plt.title("Moyenne par audio du ZCR")
    plt.xlabel("Audio")
    plt.ylabel("ZCR moyen")
    
    
    
    
    
    
    
    
    

