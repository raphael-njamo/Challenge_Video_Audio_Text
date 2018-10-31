
# coding: utf-8

# ## Import des librairies

# In[1]:


import random
import numpy as np
import numpy.linalg as la
import numpy as np
import itertools
import csv
import time
import os
from os import listdir
import scipy
import sklearn.mixture as skm
from math import *
from numpy import append, zeros
from scipy.io import wavfile
from scipy import linalg
import librosa
import pandas as pd
from tqdm import tqdm

# ## Récupération des fichiers .wav

# In[2]:


REP='data/audio/'
lins=listdir(REP)
lins.sort()
lins = [x for x in lins if x[-4:]=='.wav']


# ## Création des matrices mfcc des fichiers .wav

# In[3]:


matriceMFCC = {}
for filename in tqdm(lins):
    matriceMFCC[filename]=librosa.feature.mfcc(wavfile.read(REP+filename)[1].astype(float),n_mfcc=13)


# ## Export des matrices mfcc en csv

# In[4]:


featuresMFCC = pd.DataFrame.from_dict(matriceMFCC, orient='index')


# In[5]:


featuresMFCC.index = [x[:7] for x in featuresMFCC.index]


# In[6]:


vectFeatures = {}
for i in tqdm(range(0,13)):
    vectFeatures[str(i)] = [featuresMFCC[0][x][i] for x in range(len(featuresMFCC[0]))]


# In[7]:


# Pour que le type liste soit prit en entier et pas des '...' à la place
np.set_printoptions(threshold=np.nan)


# In[8]:


for i in tqdm(vectFeatures):
    dt = pd.DataFrame({'name':featuresMFCC.index,'mfcc':vectFeatures[i]})
    dt = dt.set_index('name')
    dt.to_csv(f'features/audio/mfcc_{i}.csv',header=False)