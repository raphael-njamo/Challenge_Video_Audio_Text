# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 14:44:20 2018

@author: linan
"""

from sklearn.mixture import GaussianMixture
import pandas as pd
from scipy.io.wavfile import read
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

#On realise les statistique descriptive sur l'energie

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
    
#----------- Test sur un audio
    sr, y= read("D:/Projet n°1/Challenge_Video_Audio_Text/data/audio/SEQ_104_AUDIO.wav")
    
    srb, yb= read("D:/projet n°1/Bruit2.0/SEQ_120_BLOND.wav")

    yb = yb[yb!=0]
    yb = yb/np.linalg.norm(yb)
    enjb = energy(yb, srb)
    
    y = y[y!=0]
    y = y/np.linalg.norm(y)
    enj = energy(y, sr)

    temps = np.cumsum([2048/sr]*len(enj))
    tempsb = np.cumsum([2048/sr]*len(enjb))
    
    tempsy = np.cumsum([1/16000]*len(y)) 
    tempsyb = np.cumsum([1/16000]*len(yb))
    
    plt.figure(1, figsize=(14, 5))
    #plt.plot(temps, enj, color="gray")
    plt.plot(tempsb, enjb, color="gray")
    plt.xlabel("Temps (s)")
    plt.ylabel("Energie")
    plt.title("Calcul de l'energie sur le bruit")
    plt.savefig('nrj.pdf')
    
    plt.figure(1, figsize=(14, 5))
    #plt.plot(tempsy, y, color="blue")
    plt.plot(tempsy, y, color="blue")
    plt.title("Signal brute")
    plt.xlabel("Temps (s)")
    plt.ylabel("Amplitude")
    plt.savefig('signal.pdf')
    plt.show()
    
    np.argmax(enj)
    max(enj)
    np.shape(enj)
    np.argmin(enj)
    min(enj)
    np.mean(enj)
    np.std(enj)
    np.median(enj)    
    
#------------------------------------------------------------------------------
# Sur tout les audios de la parole
    path = "D:/Audio2.0/"
    path2 = "D:/projet n°1/Bruit2.0/"
    infiles = [path2 + file for file in os.listdir(path2) if file.split(".")[-1]=="wav"]
    

    nrg=[]
    moye = []
    maxim = []
    ecart = []
    #stat = pd.DataFrame()
    #Sequence_decoupe = pd.DataFrame()
    for file in infiles:
        sr, y = read(file)
        yc = y[y!=0]
        nrj = energy(yc, sr)
        moy = np.mean(nrj)
        maxi = np.max(nrj)
        ecrt = np.std(nrj)
       # dec = decoupage(nrj)
        nrg.append(nrj)
        moye.append(moy)
        maxim.append(maxi)
        ecart.append(ecrt)
        
    np.argmax(maxim)
    np.argmax(ecart)
    
        Sequence_decoupe = Sequence_decoupe.append({"Sequence":file[12:-14], "Moyenne1":dec[0][0], "Moyenne2":dec[0][1], "Moyenne3":dec[0][2], "Ecart-type1":dec[1][0], "Ecart-type2":dec[1][1], "Ecart-type3":dec[1][2], "Mediane1":dec[2][0],"Mediane2":dec[2][1],"Mediane3":dec[2][2], "Max1":dec[3][0],"Max2":dec[3][1],"Max3":dec[3][2], "Min1":dec[4][0],"Min2":dec[4][1],"Min3":dec[4][2]},ignore_index=True)
    #Sequence_decoupe.to_csv("Feature2_decoup_parole.csv", sep="§", index=False) 
        argmaxi = np.argmax(nrj)
        maxi = max(nrj)
        argmini = np.argmin(nrj)
        mini = min(nrj)
        moy = np.mean(nrj)
        ecartt = np.std(nrj)
        mediane = np.median(nrj)
        long_sign = len(y)
        long_parole = len(yc)
        
        stat = stat.append({"Sequence":file[12:-14],"Long_s":long_sign,"Long_p":long_parole ,"Argmax":argmaxi ,"Max":maxi ,"Argmin":argmini ,"Min":mini ,"Moyenne":moy ,"Mediane":mediane ,"Ecart-type":ecartt}, ignore_index=True)
    #stat.to_csv("Feature_stat_parole.csv", sep="§", index=False)
        tx = stat["Long_p"]/stat["Long_s"]
        np.mean(tx)
        plt.scatter(tx.index, tx)
        plt.hist(stat["Ecart-type"], normed=1, color = 'aquamarine', edgecolor = 'black')
        pyplot.xlabel('valeurs')
        pyplot.ylabel('nombres')
        pyplot.title('Exemple d\' histogramme simple')
#taille de l'ech
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    