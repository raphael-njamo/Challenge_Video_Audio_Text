# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 13:38:51 2018

@author: linan
"""

from Yin import yin
import scipy

#voix entre 100 et 300 
(sr,sig)=scipy.io.wavfile.read("D:/Challenge_Video_Audio_Text/data/audio/SEQ_001_AUDIO.wav")

#harmo_thresh = seuil en dessous c'est de la parole
pitche, harmonic_rate, argmins, times = yin.compute_yin(sig, sr, dataFileName=None, w_len=4096, w_step=2048, f0_min=80, f0_max=300, harmo_thresh=0.5)

plt.plot(pitche, "rx")
plt.show()
plt.plot(harmonic_rate)
#frequence fond = cordes vocales, formant = frequences de raisonnance de conduit vocal -> caracterise bien les voyelles
 

















