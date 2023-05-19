# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 10:41:20 2022

@author: romas
"""

import time as tm

import sklearn.linear_model as sklm

import sklearn.tree as skt

import numpy as np


def train_evaluate_model(model: sklm._base.LinearRegression|skt._classes.DecisionTreeClassifier, X: np.ndarray, Y: np.ndarray, train: np.ndarray, test: np.ndarray, ind: int) -> tuple:
    
    X_train, Y_train = X[train], Y[train]

    X_test, Y_test = X[test], Y[test]

    start = tm.perf_counter()
    
    model.fit(X_train, Y_train)

    time = tm.perf_counter()-start

    print(f"Цикл {ind}:")
    
    print(f"Час навчання {time}")
    
    print()

    prediction = model.predict(X_test)

    return X_test, Y_test, prediction


