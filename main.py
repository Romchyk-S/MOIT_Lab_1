# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 11:26:16 2022

@author: romas
"""

import pandas as pd

import sklearn.tree as skt

import numpy as np

import sklearn.model_selection as skms

import sklearn.linear_model as sklm

# import sklearn.metrics as skm

import matplotlib.pyplot as plt

import time as tm


print()


def model_fitting(model, kf, X, Y):

    scores = []

    for train, test in kf.split(X):

        X_train = X[train]

        X_test = X[test]

        Y_train = Y[train]

        Y_test = Y[test]

        X_train, X_test, Y_train, Y_test = skms.train_test_split(X, Y, random_state = 3)

        start = tm.time()

        model.fit(X_train, Y_train)

        print(f"Час навчання {tm.time()-start}")

        print()

        scores.append(model.score(X_test, Y_test))

        # prediction = model.predict(X_test)

        # fig1 = plt.figure()

        # ax1 = fig1.subplots()

        # ax1.scatter(X_test.T[0], Y_test)

        # ax1.scatter(X_test.T[0], prediction)

    print(np.mean(scores))

    print()



dataset = pd.read_csv('weather.csv')

dataset = dataset.dropna()


X = dataset[['MinTemp', 'MaxTemp', 'Evaporation', 'Sunshine']].values

Y = dataset['Rainfall'].values



kf = skms.KFold(n_splits = 5, shuffle = True)

model = sklm.LinearRegression()

model_fitting(model, kf, X, Y)



# prediction = model.predict(X_test);

# precision = skm.precision_score(Y_test, prediction)

# recall = skm.recall_score(Y_test, prediction)

# print(f"Влучність (precision): {precision*100}%")

# print(f"Відкликання/чутливість (recall/sensitivity): {recall*100}%")

