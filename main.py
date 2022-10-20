# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 11:26:16 2022

@author: romas
"""

import pandas as pd

import sklearn.tree as skt

import sklearn.model_selection as skms

import matplotlib.pyplot as plt

import build_model as bm



import seaborn as sns

# import tkinter as tk


print()

var_to_predict = 'Rainfall'

# var_to_predict = 'Sunshine'

# var_to_predict = 'MinTemp'


corr_threshold = [0.2, 1]

splits_number = 5


dataset = pd.read_csv('weather.csv')

dataset = dataset.dropna()

corr = dataset.corr()

var_correlation = dict(corr[var_to_predict])

# best_var_correlation = {k: v for k, v in var_correlation.items() if v < corr_threshold[1] and v > corr_threshold[0]}

best_var_correlation = {k: v for k, v in var_correlation.items() if abs(v) < corr_threshold[1] and abs(v) > corr_threshold[0]}

print(best_var_correlation)

print()



corr.style.background_gradient(cmap = 'coolwarm')

sns.heatmap(corr, annot = True)


X = dataset[best_var_correlation.keys()].values

Y = dataset[var_to_predict].values



kf = skms.KFold(n_splits = splits_number, shuffle = True)


print("Лінійна регресія")

bm.build_model(kf, X, Y, 1)

print("Квадратурна регресія")

bm.build_model(kf, X, Y, 2)


# print("Дерево прийняття рішень")

# param_grid = {"max_depth": [5, 15, 25], "min_samples_leaf": [1, 3], "max_leaf_nodes": [10, 20, 35, 50]}

# model = skt.DecisionTreeClassifier()

# model = skms.GridSearchCV(model, param_grid, scoring = "f1", cv = 5)

# model_fitting(model, kf, X, Y)

