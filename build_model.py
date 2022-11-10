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

import sklearn.tree as skt

# import sklearn.model_selection as skms

import graphviz as gphv

import os

os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'



def train_evaluate_model(model, X, Y, train, test, ind):

    X_train, Y_train = X[train], Y[train]

    X_test, Y_test = X[test], Y[test]

    start = tm.perf_counter()

    model.fit(X_train, Y_train)

    time = tm.perf_counter()-start

    print(f"Цикл {ind}: Час навчання {time}")

    prediction = model.predict(X_test)

    return X_test, Y_test, prediction



def build_regression_model(kf, X, Y, regression_degree):

    poly = skp.PolynomialFeatures(degree = regression_degree, include_bias = False)

    X = poly.fit_transform(X)

    scores = []

    errs = []
    
    i = 0

    for train, test in kf.split(X):

        model = sklm.LinearRegression()

        X_test, Y_test, prediction = train_evaluate_model(model, X, Y, train, test, i)

        model_performance = model.score(X_test, Y_test)

        error = skm.mean_squared_error(Y_test, prediction)

        scores.append(model_performance)

        errs.append(error)
        
        i += 1

    # print(model.coef_)

    print(f"Середня точність за {kf.n_splits} поділів: {round(np.mean(scores), 3)*100}%")

    print(f"Середня похибка за {kf.n_splits} поділів: {round(np.mean(errs), 3)}")

    print()

def build_decision_tree_model(kf, X, Y, features, tree_parameters):

    # param_grid = {"max_depth": [5, 15, 25], "min_samples_leaf": [1, 3], "max_leaf_nodes": [10, 20, 35, 50]}

    accuracies = []

    precisions = []

    recalls = []
    
    i = 0

    for train, test in kf.split(X):    

        model = skt.DecisionTreeClassifier(max_depth=tree_parameters.get('max_depth', 5), min_samples_leaf=tree_parameters.get('min_samples_leaf', 2), max_leaf_nodes=tree_parameters.get('max_leaf_nodes', 10))

        # gs = skms.GridSearchCV(model, param_grid, scoring = "f1", cv = 5)

        X_test, Y_test, prediction = train_evaluate_model(model, X, Y, train, test, i)

        accuracy = skm.accuracy_score(Y_test, prediction)

        precision = skm.precision_score(Y_test, prediction)

        recall = skm.recall_score(Y_test, prediction)

        accuracies.append(accuracy)

        precisions.append(precision)

        recalls.append(recall)
        
        i += 1

    print()

    # print(gs.best_params_)

    print(f"Точність (accuracy): {np.mean(accuracies)*100}%")

    print(f"Влучність (precision): {np.mean(precisions)*100}%")

    print(f"Відкликання/чутливість (recall/sensitivity): {np.mean(recalls)*100}%")

    print()
    
    dot_file = skt.export_graphviz(model, feature_names=features)

    graph = gphv.Source(dot_file)

    graph.render(filename="tree", format = "png", cleanup=True)
    
