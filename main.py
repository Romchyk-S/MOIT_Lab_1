# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 11:26:16 2022

@author: romas
"""

import pandas as pd

import sklearn.tree as skt

import numpy as np

import sklearn.preprocessing as skp

import sklearn.model_selection as skms

import sklearn.linear_model as sklm

import sklearn.metrics as skm

import matplotlib.pyplot as plt

import time as tm

import seaborn as sns


print()


def model_fitting(model, kf, X, Y):

    scores = []

    for train, test in kf.split(X):

        X_train, Y_train = X[train], Y[train]

        X_test, Y_test = X[test], Y[test]

        start = tm.time()

        model.fit(X_train, Y_train)

        time = tm.time()-start

        print(model.coef_)

        print(f"Час навчання {time}")

        prediction = model.predict(X_test)

        model_performance = model.score(X_test, Y_test)

        print(model_performance)

        model_performance = skm.mean_squared_error(Y_test, prediction)

        print(model_performance)

        print()

        scores.append(model_performance)

    print(f"Середня точність за {kf.n_splits} поділів: {np.mean(scores)}")

    # print(f"Середня точність за {kf.n_splits} поділів: {round(np.mean(scores), 3)*100}%")

    print()



dataset = pd.read_csv('weather.csv')

dataset = dataset.dropna()


# X = dataset[['MinTemp', 'MaxTemp', 'Evaporation', 'Sunshine']].values

X = dataset[['MinTemp', 'MaxTemp']].values

# X = dataset[['MinTemp', 'MaxTemp', 'Evaporation']].values

Y = dataset['Rainfall'].values


corr = dataset.corr()

print(dict(corr['Rainfall']))

corr.style.background_gradient(cmap='coolwarm')

sns.heatmap(corr, annot = True)



# kf = skms.KFold(n_splits = 5, shuffle = True)

# print("Лінійна регресія")

# model = sklm.LinearRegression()

# model_fitting(model, kf, X, Y)


# print("Квадратурна регресія")

# poly = skp.PolynomialFeatures(degree = 2, include_bias = False)

# poly_features = poly.fit_transform(X)


# model = sklm.LinearRegression()

# model_fitting(model, kf, poly_features, Y)


# print("Дерево прийняття рішень")

# param_grid = {"max_depth": [5, 15, 25], "min_samples_leaf": [1, 3], "max_leaf_nodes": [10, 20, 35, 50]}

# model = skt.DecisionTreeClassifier()

# model = skms.GridSearchCV(model, param_grid, scoring = "f1", cv = 5)

# model_fitting(model, kf, X, Y)


# prediction = model.predict(X_test);

# precision = skm.precision_score(Y_test, prediction)

# recall = skm.recall_score(Y_test, prediction)

# print(f"Влучність (precision): {precision*100}%")

# print(f"Відкликання/чутливість (recall/sensitivity): {recall*100}%")

