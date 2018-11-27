# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 15:27:53 2018

@author: linan
"""

import numpy as np
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
from scipy import signal

#On test les fonctions de base pour visualiser simplement un signal
(fe,signal)=read("D:/Challenge_Video_Audio_Text/data/audio/SEQ_103_AUDIO.wav")

#on normalise le signal
signal = signal/np.linalg.norm(signal)
#fe = frequence echantillonage

plt.plot(signal)
plt.figure()
f,t,Sxx = scipy.signal.spectrogram(signal,fe)
plt.pcolormesh(t,f,np.log(Sxx), cmap='gray') #☻Spectra à la place de 'gray', gris = bruit de fond, il faut chercher a débruiter


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



if __name__ == "__main__" :
    sr, y= read("D:/Challenge_Video_Audio_Text/data/audio/SEQ_001_AUDIO.wav")
    enj = energy(y, sr)
    temps = np.cumsum([2048/sr]*len(enj))
    plt.plot(temps, enj)
    plt.show()

    
    b, a = signal.butter(3, 0.15) 
    filt = signal.filtfilt(b, a, enj)
    plt.plot(filt)
    plt.axhline(np.mean(filt)) #♥*0.95
    
#signaltonoise (SNR) évaluer la qualité de l'enregistrement 
#evalue la parole et la non parole

    
    
    






