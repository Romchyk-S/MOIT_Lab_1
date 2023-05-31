# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 13:11:17 2022

@author: romas
"""

import pandas as pd

import numpy as np


def prepare_dataset():
    
    dataset = pd.read_csv('weather.csv').dropna()
    
    variables = dataset.columns

    continuous_vars = [var for var in variables if dataset.dtypes[var] != "object"]

    discrete_vars = np.setdiff1d(variables, continuous_vars)

    
    for dv in dataset[discrete_vars]:
        
        if dataset[dv][0] in ['Yes', 'No']:
        
            dataset[dv+"_int"] = dataset[dv] == "Yes"
            
            dataset = dataset.drop(dv, axis=1)
            
        else:
            
            wind_set = set(dataset[dv])
            
            wind_dict_reversed = dict(enumerate(wind_set))
            
            wind_dict = {v:k for k,v in wind_dict_reversed.items()}
            
            dataset[dv+"_int"] = list(map(lambda x: wind_dict.get(x), dataset[dv]))
            
            dataset = dataset.drop(dv, axis=1)
        
    return dataset, continuous_vars, discrete_vars