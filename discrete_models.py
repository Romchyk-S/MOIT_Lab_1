# -*- coding: utf-8 -*-
"""
Created on Fri May 19 12:58:50 2023

@author: romas
"""

import numpy as np

import sklearn.metrics as skm

import sklearn.ensemble as ske

import sklearn.model_selection as skms

import sklearn.tree as skt

import graphviz as gphv

import os

import model_training as mt

os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

def build_decision_tree_model(kf: skms._split.KFold, X: np.ndarray, Y: np.ndarray, features: list, tree_parameters: dict):

    print("Дерево прийняття рішень")

    # param_grid = {"max_depth": [5, 15, 25], "min_samples_leaf": [1, 3], "max_leaf_nodes": [10, 20, 35, 50]}

    accuracies, precisions, recalls = [], [], []
    
    i = 0

    for train, test in kf.split(X):    

        model = skt.DecisionTreeClassifier(max_depth=tree_parameters.get('max_depth', 5), min_samples_leaf=tree_parameters.get('min_samples_leaf', 2), max_leaf_nodes=tree_parameters.get('max_leaf_nodes', 10), random_state = tree_parameters.get('random_state', 666))

        X_test, Y_test, prediction = mt.train_evaluate_model(model, X, Y, train, test, i)

        accuracy = skm.accuracy_score(Y_test, prediction)
        
        try:
            
            precision = skm.precision_score(Y_test, prediction)

            recall = skm.recall_score(Y_test, prediction)
            
        except ValueError:
            
            precision = skm.precision_score(Y_test, prediction, average = 'micro')
    
            recall = skm.recall_score(Y_test, prediction, average = 'micro')

        accuracies.append(accuracy)

        precisions.append(precision)

        recalls.append(recall)
        
        i += 1

    print()
    
    print(f"Точність (accuracy): {np.mean(accuracies)*100}%")

    print(f"Влучність (precision): {np.mean(precisions)*100}%")

    print(f"Відкликання/чутливість (recall/sensitivity): {np.mean(recalls)*100}%")

    print()
    
    dot_file = skt.export_graphviz(model, feature_names=features)

    graph = gphv.Source(dot_file)

    graph.render(filename="tree", format = "png", cleanup=True)
    
def build_random_forest_model(kf: skms._split.KFold, X: np.ndarray, Y: np.ndarray, features: list, forest_parameters: dict):
    
    print("Випадковий ліс")
    
    accuracies, precisions, recalls = [], [], []
    
    i = 0
    
    for train, test in kf.split(X): 
        
        model = ske.RandomForestClassifier(max_features = forest_parameters.get('max_features', 5), n_estimators = forest_parameters.get('n_estimators', 10), random_state = forest_parameters.get('random_state', 666))

        X_test, Y_test, prediction = mt.train_evaluate_model(model, X, Y, train, test, i)

        accuracy = skm.accuracy_score(Y_test, prediction)
        
        try:
            
            precision = skm.precision_score(Y_test, prediction)

            recall = skm.recall_score(Y_test, prediction)
            
        except ValueError:
            
            precision = skm.precision_score(Y_test, prediction, average = 'micro')
    
            recall = skm.recall_score(Y_test, prediction, average = 'micro')
            
        
        confusion_matr = skm.confusion_matrix(Y_test, prediction)
        
        print("Confusion matrix:")

        print(confusion_matr)

        accuracies.append(accuracy)

        precisions.append(precision)

        recalls.append(recall)
        
        i += 1

    
    print()
    
    print(f"Точність (accuracy): {np.mean(accuracies)*100}%")

    print(f"Влучність (precision): {np.mean(precisions)*100}%")

    print(f"Відкликання/чутливість (recall/sensitivity): {np.mean(recalls)*100}%")

    print()

    i = 0
    
    for tree in model:
    
        dot_file = skt.export_graphviz(tree, feature_names=features)
    
        graph = gphv.Source(dot_file)
    
        graph.render(filename=f"forest_{i}", format = "png", cleanup=True)
        
        i += 1
