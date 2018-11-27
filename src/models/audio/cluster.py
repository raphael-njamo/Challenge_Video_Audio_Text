# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 10:03:36 2018

@author: linan
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
import sklearn
from librosa.display import waveplot
from librosa.onset import onset_detect
from librosa import frames_to_time
from librosa import frames_to_samples
from librosa import load
from librosa import zero_crossings
from mir_eval.sonify import clicks
from IPython.display import Audio
#from urllib import urlretrieve

plt.rcParams['figure.figsize'] = (14, 4)

path = ""
filename = "D:/ALL_SEQ2.wav"
#urlretrieve('http://audio.musicinformationretrieval.com/' + filename, filename=filename)

#Audio(filename)

x, fs = load(filename)
#print (fs)

#waveplot(x,fs)

onset_frames = onset_detect(x, sr=fs, delta=0.04, wait=4)
onset_times = frames_to_time(onset_frames, sr=fs)
onset_samples = frames_to_samples(onset_frames)

x_with_beeps = clicks(onset_times, fs, length=len(x))

#Audio(x + x_with_beeps, rate=fs)

#feature extraction
def extract_features(x, fs):
    zcr = zero_crossings(x).sum()
    energy = scipy.linalg.norm(x)
    return [zcr, energy]

frame_sz = fs*0.090
features = np.array([extract_features(x[int(i):int(i+frame_sz)], fs) for i in onset_samples])
print (features.shape)

#feature scaling
min_max_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))
features_scaled = min_max_scaler.fit_transform(features)
print (features_scaled.shape)
print (features_scaled.min(axis=0))
print (features_scaled.max(axis=0))

plt.scatter(features_scaled[:,0], features_scaled[:,1])
plt.xlabel('Zero Crossing Rate (scaled)')
plt.ylabel('Spectral Centroid (scaled)')











