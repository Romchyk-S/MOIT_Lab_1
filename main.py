# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 11:26:16 2022

@author: romas
"""

import prepare_data as prd

# import seaborn as sns

import graphic_interface as gi


print()

# інтервал кореляції для включення незалежної змінної в модель, відкритий із обох боків
corr_threshold = [0.2, 0.6] 

# кількість різних поділів масиву даних
splits_number = 5

# параметри для побудови дерева
tree_parameters = {'max_depth': 5, 'min_samples_leaf': 2, 'max_leaf_nodes': 10}

# максимальний степінь регресії
max_regression_pow = 2


dataset, continuous_vars, discrete_vars = prd.prepare_dataset()

        
# corr = dataset.corr()
    
# corr.style.background_gradient(cmap = 'coolwarm')

# sns.heatmap(corr, annot = True)


gi.main_work(dataset, continuous_vars, discrete_vars, corr_threshold, splits_number, max_regression_pow, tree_parameters)

