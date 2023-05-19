# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 12:07:05 2022

@author: romas
"""

import pandas as pd

import numpy as np

import customtkinter as ctk

import sklearn.model_selection as skms

import continuous_models as cms

import discrete_models as dms


def main_work(dataset: pd.core.frame.DataFrame, continuous_vars: list[str], discrete_vars: np.ndarray, corr_threshold: list[float], splits_number: int, tree_parameters: dict) -> None:

    root = ctk.CTk()

    ctk.set_appearance_mode("System")

    ctk.set_default_color_theme("dark-blue")
    
    root.title("Вибір змінних")

    root.geometry('700x700')

    choose_continuous_var(root, dataset, continuous_vars, corr_threshold, splits_number)

    choose_discrete_var(root, dataset, discrete_vars, corr_threshold, splits_number, tree_parameters)

    end_window(root)    

    root.mainloop()
    
def end_window(root: ctk.windows.ctk_tk.CTk) -> None:

     def button():

         root.destroy()

     submit_button = ctk.CTkButton(root, text = 'Завершити роботу', command = lambda: button())
     submit_button.pack()

def get_independent_vars(dataset: pd.core.frame.DataFrame, var_to_predict: str, corr_threshold: list[float], var_type: str) -> dict:
    
    corr = dataset.corr()
    
    if var_type == "continuous":

        var_correlation = dict(corr.get(var_to_predict))
        
    elif var_type == "discrete":
        
        var_correlation = dict(corr.get(var_to_predict+"_int"))

    best_var_correlation = {k: v for k, v in var_correlation.items() if abs(v) < corr_threshold[1] and abs(v) > corr_threshold[0]}
    
    
    while len(best_var_correlation) == 0:
        
        corr_threshold[0] -= 0.1
        
        corr_threshold[1] += 0.1
        
        best_var_correlation = {k: v for k, v in var_correlation.items() if abs(v) < corr_threshold[1] and abs(v) > corr_threshold[0]}

    
    return best_var_correlation

def prepare_model_parameters(dataset: pd.core.frame.DataFrame, var_to_predict: str, corr_threshold: list[float], splits_number: int,  var_type: str) -> tuple:
    
    best_var_correlation = get_independent_vars(dataset, var_to_predict, corr_threshold, var_type)
    
    X = dataset[best_var_correlation.keys()].values
    
    if var_type == "continuous":

        Y = dataset[var_to_predict].values
        
    elif var_type == "discrete":
        
        Y = dataset[var_to_predict+"_int"].values        
    
    kf = skms.KFold(n_splits = splits_number, shuffle = True)
    

    return best_var_correlation, X, Y, kf

def choose_continuous_var(root: ctk.windows.ctk_tk.CTk, dataset: pd.core.frame.DataFrame, continuous_vars: list[str], corr_threshold: list[float], splits_number: int) -> None:

    # print(dataset.columns)

    label = ctk.CTkLabel(root, text = "Оберіть максимальний степінь регресії: ")

    label.pack()

    value_inside = ctk.IntVar(value = 2)
    
    textbox = ctk.CTkEntry(root, textvariable = value_inside)
    
    textbox.pack()
    

    label_1 = ctk.CTkLabel(root, text = "Оберіть неперервну змінну: ")

    label_1.pack()

    value_inside_1 = ctk.StringVar()

    menu = ctk.CTkOptionMenu(root, variable = value_inside_1, values = continuous_vars)

    menu.pack()


    def get_cont_var():
        
        max_regression_pow = value_inside.get()
        

        cont_var_to_predict = value_inside_1.get()

        print(f"Обрано змінну {cont_var_to_predict}")

        best_cont_var_correlation, X, Y, kf = prepare_model_parameters(dataset, cont_var_to_predict, corr_threshold, splits_number, "continuous")

        print(f"Обрана змінна найкраще корелює зі змінними: {best_cont_var_correlation}")

        print()
        
        regression_powers = [i for i in range(1, max_regression_pow+1)]
        
        regression_names = {1: "Лінійна регресія", 2: "Квадратурна регресія", 3: "Кубічна регресія"}
        
        for rp in regression_powers:

            print(regression_names.get(rp, f"Регресія степеня {rp}"))

            cms.build_regression_model(kf, X, Y, rp)
            
        cms.build_neural_network(kf, X, Y)


    submit_button = ctk.CTkButton(root, text='Обрати змінну', command = lambda: get_cont_var())

    submit_button.pack()

def choose_discrete_var(root: ctk.windows.ctk_tk.CTk, dataset: pd.core.frame.DataFrame, discrete_vars: np.ndarray, corr_threshold: list[float], splits_number: int, tree_parameters: dict) -> None:

    label_1 = ctk.CTkLabel(root, text="Оберіть дискретну змінну: ")

    label_1.pack()

    value_inside_1 = ctk.StringVar()

    cond_entry_1 = ctk.CTkOptionMenu(root, variable = value_inside_1, values = discrete_vars)

    cond_entry_1.pack()


    def get_discr_var():

        discrete_var_to_predict = value_inside_1.get()

        print(f"Обрано змінну {discrete_var_to_predict}")
        

        best_discrete_var_correlation, X, Y, kf = prepare_model_parameters(dataset, discrete_var_to_predict, corr_threshold, splits_number, "discrete")
        
        print(f"Обрана змінна найкраще корелює зі змінними: {best_discrete_var_correlation}")

        print()
        
        dms.build_decision_tree_model(kf, X, Y, list(best_discrete_var_correlation), tree_parameters)
        

    submit_button = ctk.CTkButton(root, text='Обрати змінну', command = lambda: get_discr_var())
    
    submit_button.pack()