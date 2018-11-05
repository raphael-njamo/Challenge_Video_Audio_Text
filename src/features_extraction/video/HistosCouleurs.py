#!/usr/bin/env python
# coding: utf-8

# ## Histogrammes de couleurs

# Import des librairies

import cv2
import os
from scipy.misc import imread
import pandas as pd
import numpy as np

# Initialisation du répertoire

REP = 'data/images'

# Dictionnaire chemin des images

lins = os.listdir(REP)
lins = [x for x in lins if '.' not in x[:]]
lins.sort()


lins2 = {}
for i in lins:
    lins2[i] = os.listdir(f'{REP}/{i}')

# Histogramme

hist = []
for i in lins2:
    for j in lins2[i]:
        hist.append(cv2.calcHist(imread(f'{REP}/{i}/{j}'), [0], None, [256], [0, 256]))

# Index

name = []
for i in lins2:
    for j in lins2[i]:
        name.append(f'{i}/{j}')

# Dataframe histogramme

dt_hist = pd.DataFrame.from_records(hist,index=name)

# Array to scalar ; ex : [158] -> 158

for i in range(256):
    for j in range(len(dt_hist[1])):
        dt_hist[i][j] = dt_hist[i][j][0]

# Dataframe to csv

dt_hist.to_csv('features/video/histogrammes.csv')

