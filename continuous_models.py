# -*- coding: utf-8 -*-
"""
Created on Fri May 19 12:56:37 2023

@author: romas
"""

import sklearn.metrics as skm

import numpy as np

import sklearn.preprocessing as skp

import sklearn.linear_model as sklm

import sklearn.model_selection as skms

import tensorflow.keras.models as tkm

import tensorflow.keras.layers as tkl

import matplotlib.pyplot as plt

import model_training as mt



def build_regression_model(kf: skms._split.KFold, X: np.ndarray, Y: np.ndarray, regression_degree: int) -> None:

    poly = skp.PolynomialFeatures(degree = regression_degree, include_bias = False)

    X = poly.fit_transform(X)

    scores, errs = [], []
    
    i = 0

    for train, test in kf.split(X):

        model = sklm.LinearRegression()

        X_test, Y_test, prediction = mt.train_evaluate_model(model, X, Y, train, test, i)

        model_performance = model.score(X_test, Y_test)

        error = skm.mean_squared_error(Y_test, prediction)

        scores.append(model_performance)

        errs.append(error)
        
        i += 1

    print(f"Середня точність за {kf.n_splits} поділів: {round(np.mean(scores), 3)*100}%")

    print(f"Середня похибка за {kf.n_splits} поділів: {round(np.mean(errs), 3)}")

    print()

def build_neural_network(kf: skms._split.KFold, X: np.ndarray, Y: np.ndarray) -> None:
    
    print("Нейронна мережа")
    
    i = 0
    
    scores, errs_test = [], []
    
    poly = skp.PolynomialFeatures(degree = 1, include_bias = False)

    X = poly.fit_transform(X)
    
    X = np.interp(X, (X.min(), X.max()), (0, +1))
    
    Y = np.interp(Y, (Y.min(), Y.max()), (0, +1))
    
    for num, train_test in enumerate(kf.split(X)):
        
        train, test = train_test
    
        model = tkm.Sequential()
        
        model.add(tkl.Dense(len(X), activation='relu'))
        
        model.add(tkl.Dense(64, activation='tanh'))
        
        model.add(tkl.Dense(32, activation='tanh'))
        
        model.add(tkl.Dense(16, activation='tanh'))
        
        model.add(tkl.Dense(8, activation='tanh'))
        
        model.add(tkl.Dense(4, activation='relu'))
        
        model.add(tkl.Dense(1))
        
        model.compile(loss = "mse", metrics=['accuracy'])

        
        plt.title(f"Мережа {num}")
    
        X_test, Y_test, prediction = mt.train_evaluate_model(model, X, Y, train, test, i)

        model_performance = model.evaluate(X_test, Y_test)
        
        errors = {round(float(k), 5):round(float(v), 5) for k, v in zip(Y_test, prediction) if k != v}
        
        print("Невідповідності в елементах між y_test та prediction")
        
        print(errors)
        
        print(f"Відсотків невідповідних: {len(errors)/len(prediction)*100}%")

        scores.append(model_performance[1])
        
        errs_test.append(model_performance[0])
        
        print()
        
        i += 1
    
    print(f"Середня точність за {kf.n_splits} поділів: {round(np.mean(scores), 3)*100}%")
    
    print(f"Середня похибка за {kf.n_splits} поділів (x_test vs y_test): {round(np.mean(errs_test), 3)}")

    print()
    
