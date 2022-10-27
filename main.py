# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 11:26:16 2022

@author: romas
"""

import pandas as pd

# import matplotlib.pyplot as plt

# import seaborn as sns

import numpy as np

import graphic_interface as gi


print()

corr_threshold = [0.2, 1]

splits_number = 5


dataset = pd.read_csv('weather.csv')

dataset = dataset.dropna()

# corr.style.background_gradient(cmap = 'coolwarm')

# sns.heatmap(corr, annot = True)



variables = dataset.columns

continuous_vars = [var for var in variables if dataset.dtypes[var] != "object"]

discrete_vars = np.setdiff1d(variables, continuous_vars)


gi.main_work(dataset, continuous_vars, discrete_vars, corr_threshold, splits_number)

