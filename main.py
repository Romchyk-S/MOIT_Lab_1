# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 11:26:16 2022

@author: romas
"""

import seaborn as sns

import matplotlib.pyplot as plt

import prepare_data as prd

import graphic_interface as gi


# make normal confusion matrix labels

print()

# інтервал кореляції для включення незалежної змінної в модель, відкритий із обох боків
corr_threshold = [0.2, 0.6] 

# кількість різних поділів масиву даних
splits_number = 5

# параметри для побудови дерева
tree_parameters = {'max_depth': 5, 'min_samples_leaf': 2, 'max_leaf_nodes': 10, 'random_state': 42}

forest_parameters = {'max_features': 5, 'n_estimators': 10, 'random_state': 42}

dataset_numerical, continuous_vars, discrete_vars = prd.prepare_dataset()
        
corr = dataset_numerical.corr()
    
corr.style.background_gradient(cmap = 'coolwarm')

sns.heatmap(corr, annot = False)

plt.show()

gi.main_work(dataset_numerical, continuous_vars, discrete_vars, corr_threshold, splits_number, tree_parameters, forest_parameters)