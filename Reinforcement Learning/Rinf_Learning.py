#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 14:18:49 2019

@author: pymacbit
"""
# Random Selection

# Importing the libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset_rs
dataset_rs = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing Random Selection 
import random
N = 10000
d = 10
ads_selected = []
total_reward = 0
for n in range(0, N):
    ad = random.randrange(d)
    ads_selected.append(ad)
    reward = dataset_rs.values[n, ad]
    total_reward = total_reward + reward

# Visualising the results - Histogram
plt.hist(ads_selected)
plt.title('Histogram of ads selection')
plt.xlabel('Ads')
plt.ylabel('Number of times each as was selected')

###########################################################################


# Importing the dataset_ucb
dataset_ucb = pd.read_csv('Ads_CTR_Optimisation.csv')


