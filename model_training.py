# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 10:41:20 2022

@author: romas
"""

import time as tm

import sklearn.linear_model as sklm

import sklearn.tree as skt

import sklearn.ensemble as skle

import numpy as np

import matplotlib.pyplot as plt

import keras as ker


def train_evaluate_model(model: sklm._base.LinearRegression|skt._classes.DecisionTreeClassifier|ker.engine.sequential.Sequential|skle._forest.RandomForestClassifier, X: np.ndarray, Y: np.ndarray, train: np.ndarray, test: np.ndarray, ind: int) -> tuple:
   
    X_train, Y_train = X[train], Y[train]

    X_test, Y_test = X[test], Y[test]

    start = tm.perf_counter()

    if type(model) == ker.engine.sequential.Sequential:
        
        history = model.fit(X_train, Y_train, epochs = 10)
        
        
        plt.plot(history.history['loss'])

        
        plt.xlabel("Епоха")
        
        plt.ylabel("Втрати")
        
        plt.show()
        
        plt.plot(history.history['accuracy'])
        
        plt.xlabel("Епоха")
        
        plt.ylabel("Точність")
        
        plt.show()
        
    else:
        
         
        model.fit(X_train, Y_train)
    
        
    time = tm.perf_counter()-start

    print(f"Цикл {ind}:")
    
    print(f"Час навчання {time}")
    
    print()
    


    prediction = model.predict(X_test)

    return X_test, Y_test, prediction

