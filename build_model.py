# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 10:41:20 2022

@author: romas
"""


import sklearn.preprocessing as skp

import sklearn.linear_model as sklm

import time as tm

import sklearn.metrics as skm

import numpy as np


def build_model(kf, X, Y, regression_degree):

    poly = skp.PolynomialFeatures(degree = regression_degree, include_bias = False)

    X = poly.fit_transform(X)

    scores = []

    errs = []

    for train, test in kf.split(X):

        model = sklm.LinearRegression()

        X_train, Y_train = X[train], Y[train]

        X_test, Y_test = X[test], Y[test]

        start = tm.time()

        model.fit(X_train, Y_train)

        time = tm.time()-start

        # print(model.coef_)

        print(f"Час навчання {time}")



        # model_performance = model.score(X_train, Y_train)

        # print("train")

        # print(model_performance*100)

        model_performance = model.score(X_test, Y_test)

        scores.append(model_performance)

        # print("test")

        # print(model_performance*100)

        prediction = model.predict(X_test)

        error = skm.mean_squared_error(Y_test, prediction)

        errs.append(error)

        # print("err")

        # print(error)

        # print()

        # i = 0

        # while i < len(prediction):

        #     print(f"{Y_test[i]}:{prediction[i]}")

        #     i += 1

        # print()

    # print(f"Середня точність за {kf.n_splits} поділів: {np.mean(scores)*100}")

    print(f"Середня точність за {kf.n_splits} поділів: {round(np.mean(scores), 3)*100}%")

    print(f"Середня похибка за {kf.n_splits} поділів: {round(np.mean(errs), 3)}")

    print()