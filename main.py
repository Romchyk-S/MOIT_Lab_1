# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 11:26:16 2022

@author: romas
"""

import pandas as pd

import sklearn.tree as skt

import sklearn.model_selection as skms

import matplotlib.pyplot as plt


dataset = pd.read_csv('weather.csv')


X = dataset[['MinTemp', 'MaxTemp', 'Evaporation']].values

Y = dataset['RainTomorrow'].values

print(Y)

fig1 = plt.figure()

ax1 = fig1.subplots()

fig2 = plt.figure()

ax2 = fig2.subplots()

ax1.scatter(dataset['Evaporation'].values, dataset['Rainfall'])

ax2.scatter(dataset['MinTemp'].values, dataset['Temp9am'])
