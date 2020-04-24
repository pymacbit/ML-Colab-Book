#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 10:04:30 2019

@author: pymacbit
"""
# Apriori

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset_ap
dataset_ap = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transations = []
for i in range(0,7501):
    transations.append([str(dataset_ap.values[i,j]) for j in range(0, 20)])
    
# Training Apriori on the dataset
from apyori import apriori
rules = apriori(transations, min_support = 0.003 , min_confidence = 0.2, min_lift = 3, min_length = 2)
















































 