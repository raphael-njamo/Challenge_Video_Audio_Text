# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 10:09:01 2018

@author: linan
"""

from sklearn.mixture import GaussianMixture
import pandas as pd
from scipy.io import wavfile
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.preprocessing import Normalizer


#pour l'enegie -> faire varier la fenetre et le pas 
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

def gaussian(x, mu, sig):
    return 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)

def seuil(enj, debug=False):
    gmm = GaussianMixture(n_components=2)
    gmm.fit(np.array(enj).reshape(len(enj),1))
    
    moyenne = gmm.means_
    sigma = gmm.covariances_
    x = np.linspace(np.min(moyenne), np.max(moyenne), 100) #le 100 a modifier en fonction de la rapidité
    xbis = np.linspace(np.min(moyenne)-4*np.min(sigma), np.max(moyenne)+4*np.min(sigma), 100)
    
    g0 = gaussian(x,moyenne[0], sigma[0][0])
    g1 = gaussian(x,moyenne[1], sigma[1][0])
    g0p = gaussian(xbis,moyenne[0], sigma[0][0])
    g1p = gaussian(xbis,moyenne[1], sigma[1][0])
#    pic0 = np.argmax(g0)
#    pic1 = np.argmax(g1)
#    if pic0>=pic1:
#        petitpic=pic1
#        gdpic = pic0
#    else:
#        petitpic=pic0
#        gdpic=pic1
#    
#    filtre = vect[petitpic:gdpic]
    
    intersection = np.argmin(np.abs(g0-g1))
    if debug:
        plt.figure(1, figsize=(14, 6))
        plt.plot(enj, color ="grey")
        
        plt.axhline(x[intersection], color="red", ls='--')
        picmax=[]
        for i in range(1,len(enj)-1):
            if enj[i-1] < enj[i] and enj[i] > enj[i+1] and enj[i]>x[intersection]:
                picmax.append(enj[i])
            else: 
                picmax.append(None)
        xmax=picmax
        picmin=[]
        for i in range(1,len(enj)-1):
            if enj[i-1] > enj[i] and enj[i] < enj[i+1] and enj[i]<x[intersection]:
                picmin.append(enj[i])
            else: 
                picmin.append(None)        
        xmin=picmin
        plt.plot(xmax, ".", color="darkblue")
        plt.plot(xmin, ".", color="coral")
        plt.legend(("Energie de l'audio", 'Seuil', 'Parole', 'Bruit'), shadow=True)
        plt.axis([-3,240,16,32.5])#non normalisé = [-5,230,15,32.5][-1,30,-11,5]
        plt.text(80,24.5,"Seuil de décision parole/bruit",{'color': 'r', 'fontsize': 12})
        plt.xticks([])
        plt.title("Détermination des zones de parole")
        plt.xlabel("Temps (s)")
        plt.ylabel("Energie")
        #gaussienne
        a = plt.axes([0.2, 0.6, .35, .23], facecolor='beige')
        plt.plot(xbis, g0p, color="coral")
        plt.plot(xbis, g1p, color="darkblue")
        plt.legend(('Bruit', 'Parole'), shadow=True)
        #plt.plot(x[intersection], np.min(np.abs(g0-g1)), ".")
        plt.title("Modèle bigaussien de la distribution de l'énergie")
        plt.savefig('dav2.pdf')
        plt.show()
        
    return x[intersection]

def decoupage(enj, nb_seq=3):
    seq = len(enj)//nb_seq
    mean = []
    std = []
    med = []
    maxi = []
    mini = []
    reste = len(enj) % nb_seq
    for i in range(nb_seq):
        seq_enj = enj[i*seq:(i+1)*seq]
        if i == nb_seq-1 and reste !=0:
            seq_enj+= enj[-reste:]
        mean.append(np.mean(seq_enj))
        std.append(np.std(seq_enj))
        med.append(np.median(seq_enj))
        maxi.append(max(seq_enj))
        mini.append(min(seq_enj))
    return mean, std, med, maxi, mini


if __name__ == "__main__" :
    #---------------------------- Test sur un audio
    sr, y = wavfile.read("D:/Challenge_Video_Audio_Text/data/audio/SEQ_088_AUDIO.wav")
    #y = y/np.linalg.norm(y)
    enj = energy(y, sr)
    seuil(enj, debug=True)    
    #on normalise le signal: pour les calcules, pas pour calculer la parole -> pas pour la frequence fond
    #enjN = enj/np.linalg.norm(enj)
    plt.plot(filt)
    b, a = signal.butter(3, 0.2) 
    filt = signal.filtfilt(b, a, enj)
    filtN = filt/np.linalg.norm(filt)
    #Pour les graphiques on ne normalise pas
    print(seuil(enj, True))
    parole =[ 0 if e<= seuil(enj) else e for e in enj]
# seuil appliqué au filtre:
    #parolefilt =[ 0 if e<= seuil(filtN) else e for e in filtN]
    #NONparole =[ 0 if e> seuil(enj) else e for e in enj]
    #plt.plot(parolefilt)
    #plt.ylabel('') -> ordonné
    #plt.axis([0, 200, 0, 0.15])
    #On calcul le seuil pour chaque audio
    
    ### DECOUPAGE DU SIGNAL 
    len(parole)
    masque = []
    for i in parole:
        if i == 0:
            masque.extend([1]* 4096)
        else:
            masque.extend([0]* 4096)
    masque=masque[:len(y)]
    Nouveau_signal = masque*y
    Nouveau_signal= np.array(Nouveau_signal, dtype=np.int16)
    write("D:/Bruit2.0/BLOND.wav",sr,Nouveau_signal)
    
    #---------------------------- Tous les audios
    path = "D:/Challenge_Video_Audio_Text/data/audio/"
    infiles = [path + file for file in os.listdir(path) if file.split(".")[-1]=="wav"]
    
    #seuils = pd.DataFrame()
    #nrjNP = []
    #parole_nonN = []
    df = pd.DataFrame()
    #Seq_decoupe = pd.DataFrame()
    #segmentation = pd.DataFrame()
    for file in tqdm(infiles):
        sr, y= wavfile.read(file) 
        enj = energy(y, sr)         
        parole = [ 0 if e<= seuil(enj) else e for e in enj]
        parole = pd.DataFrame(parole)
        longueur = len(parole)
        par = (parole !=0).sum(axis=1).sum()
        tx_parole = par/longueur
        df =df.append({"Sequence":file[41:-10],"Taux de parole":tx_parole},ignore_index=True)
    df.to_csv("Taux_Parole.csv", sep="§", index=False)
        ###### Stat sur la parole
        parole = [ 0 if e<= seuil(enj) else e for e in enj]
        #######[i for i in enj if i > s]
        #parole.append(parol)
        #######decoupage parole
        #decoup_parole = decoupage(parol)
        #Seq_decoupe = Seq_decoupe.append({"Sequence":file[41:-4], "Moyenne1":dec[0][0], "Moyenne2":dec[0][1], "Moyenne3":dec[0][2], "Ecart-type1":dec[1][0], "Ecart-type2":dec[1][1], "Ecart-type3":dec[1][2], "Mediane1":dec[2][0],"Mediane2":dec[2][1],"Mediane3":dec[2][2], "Max1":dec[3][0],"Max2":dec[3][1],"Max3":dec[3][2], "Min1":dec[4][0],"Min2":dec[4][1],"Min3":dec[4][2]},ignore_index=True)
        
        ##########################################################################
        #Exportation des signaux coupé des blonds
        masque = []
        for i in parole:
            if i == 0:
                masque.extend([1]* 2048)
            else:
                masque.extend([0]* 2048)
        masque=masque[:len(y)]
        Nouveau_signal = masque*y
        Nouveau_signal= np.array(Nouveau_signal, dtype=np.int16)
        #plt.plot(Nouveau_signal)
        #plt.plot(y[100:6000])
        write("D:/Bruit2.0/"+file[41:-9]+"BLOND.wav",sr,Nouveau_signal)


        
        ###########################################################################
        
   # ____________DECOUPAGE des audios en 3 sequences
        dec = decoupage(enj)
        Sequence_decoupe = Sequence_decoupe.append({"Sequence":file[41:-4], "Moyenne1":dec[0][0], "Moyenne2":dec[0][1], "Moyenne3":dec[0][2], "Ecart-type1":dec[1][0], "Ecart-type2":dec[1][1], "Ecart-type3":dec[1][2], "Mediane1":dec[2][0],"Mediane2":dec[2][1],"Mediane3":dec[2][2], "Max1":dec[3][0],"Max2":dec[3][1],"Max3":dec[3][2], "Min1":dec[4][0],"Min2":dec[4][1],"Min3":dec[4][2]},ignore_index=True)
    #Sequence_decoupe.to_csv("Decoupage_Sequence.csv", sep="§", index=False)
        
        #_____________ NORMALISATION -> voir pour normaliser plutot sur la f0
        #on normalise pas pour le frequence 
        #enjN = enj/np.linalg.norm(enj)
        
        #______________ SEUIL -> parole/non parole
        #seuils = seuils.append({"Sequence":file[41:-4],"Seuils":seuil(enjN)},ignore_index=True)
        #Séparation parole / non parole
        parole = [ 0 if e<= seuil(enj) else e for e in enj]
        longueur = len(parole)
        parole = (f0 !=0).sum(axis=1).sum()
        tx_parole = parole/longueur
        parole.append(parole)
        #NONparole = [ 0 if e> seuil(enjN) else e for e in enjN]
        
        #Segmentation pour indiquer si il s'agit d'un monologue ou d'un echange rapide
        seg=[]
        taille = 0
        for i in parole:
            if i != 0:
                taille += 1
            else:
                seg.append(taille)
                taille = 0
        seg.append(taille)
        #nrjNP.append(parole)
        #segmentation = segmentation.append({"Sequence":file[41:-4],"Segmentation_parole":seg},ignore_index=True)
        #segm = list(filter(lambda val: val != 0,seg)) #si je veux enlever les 0 = 
    #segmentation.to_csv("Segmentation_parole.csv", sep="§", index=False)

    
    
    
    
    
    
    
    
    
    

    







