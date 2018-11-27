# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 10:05:22 2018

@author: linan
"""

import pandas as pd
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
import os
import librosa
from tqdm import tqdm
import numpy as np

#Frequence d'echantillonage
fe = 16000

path = "D:/F0"
infiles = [path + file for file in os.listdir(path) if file.split(".")[-1]=="f0"]


#On récupére la frequence fond
Freq_fond = pd.read_csv("./F0/SEQ_001_AUDIO.f0",sep="/n")
Freq_fond.index = range(1, len(Freq_fond)+1)

plt.plot(Freq_fond)
np.shape(Freq_fond)

#calcul de features
mean_wav = {}
for files in tqdm(infiles) : 
    para = []
    freqfond = pd.read_csv(files)
    sr=16000
    
    y= y.astype(float)
    #mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
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

len(Freq_fond)
#Freq_fond.astype(bool).sum(axis=0)
(Freq_fond !=0).sum(axis=0).sum()
#axis = 1 colonne

def taux_parole(f0):
    
    longueur = len(f0)
    parole = (f0 !=0).sum(axis=1).sum()
    tx_parole = parole/longueur
    
    return tx_parole


    
if __name__ == "__main__" :
    #_____________________________ Test sur un seul audio
    f0 = pd.read_csv("D:/F0/SEQ_036_AUDIO.f0", sep="/n")  
    sr, y = wavfile.read("D:/Challenge_Video_Audio_Text/data/audio/SEQ_036_AUDIO.wav")
    plt.figure(1, figsize=(14, 6))
    y = y/np.linalg.norm(y)
    tempsy = np.cumsum([1/16000]*len(y))
    plt.axis([0,15,-0.015,0.015])
    plt.plot(tempsy, y, color="blue")    
    plt.title("Amplitude du signal")
    plt.xlabel("Temps (s)")
    plt.ylabel("Amplitude")    
    
    tps = np.cumsum([100/16000]*len(f0))
    plt.figure(1, figsize=(14, 6))
    plt.plot(tps,f0, "bx", color="indigo")#[100:2000]
    plt.axis([0,15,95,300])
    plt.xlabel("Temps (s)")
    plt.ylabel("Fréquence (Hz)")
    plt.title("Fréquence fondamentale")
    
    
    picmax=[]
    for i in range(1,len(f0)-1):
        if f0[i-1] < f0[i] and f0[i] > f0[i+1] and f0[i]>50:
            picmax.append(f0[i])
        else: 
            picmax.append(None)
    xmax=picmax
   
    picmin=[]
    for i in range(1,len(f0)-1):
        if f0[i-1] > f0[i] and f0[i] < f0[i+1] and f0[i]>50:
            picmin.append(f0[i])
        else: 
            picmin.append(None)        
    xmin=picmin
    
    plt.plot(pmax)


    list_f0 = f0['0.0'].tolist() #-> passage dataframe en list
    
    #Passage en demi-ton des HERTZ
        seg=[]
        taille = 0
        dep = 0
        for i in range(len(list_f0)):
            if list_f0[i] != 0: 
                taille += 1
                if dep ==0:
                    dep = i
            else:
                seg.append(list_f0[dep : dep+taille])
                taille = 0
                dep = 0
        
        seg = [x for x in seg if x]
        seg =  [[12*np.log2(x/(np.mean(line))+1) for x in line] for line in seg]

    lst = []    
    for s in seg:
        lst.extend(s)
    plt.plot(lst)
    plt.show()
    #print(taux_parole(f0))
    
    #____________________________ Tous les audios
    path = "D:/F0/"
    infiles = [path + file for file in os.listdir(path) if file.split(".")[-1]=="f0"]
    
    df = pd.DataFrame()
    for file in tqdm(infiles):
        f0=pd.read_csv(file,sep="/n")
        df =df.append({"Sequence":file[6:-9],"Taux de parole":taux_parole(f0)},ignore_index=True)
#        df["Sequence"]= file[5:-9]
#        df["Taux de parole"]= taux_parole(f0)
    df.to_csv("Taux_Parole.csv", sep="§", index=False)
        
        
    















