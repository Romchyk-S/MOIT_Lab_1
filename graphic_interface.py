# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 12:07:05 2022

@author: romas
"""

import tkinter as tk

import sklearn.model_selection as skms

import build_model as bm


def main_work(dataset, continuous_vars, discrete_vars, corr_threshold, splits_number, max_regression_pow, tree_parameters):

    root = tk.Tk()

    root.title("Вибір змінних")

    root.geometry('700x500')

    choose_continuous_var(root, dataset, continuous_vars, corr_threshold, max_regression_pow, splits_number)

    choose_discrete_var(root, dataset, discrete_vars, corr_threshold, splits_number, tree_parameters)

    end_window(root)    

    root.mainloop()
    
    

def end_window(root):

     def button():

         root.destroy()

     submit_button = tk.Button(root, text = 'Завершити роботу', command = lambda: button())
     submit_button.pack()


def get_independent_vars(dataset, var_to_predict, corr_threshold, var_type):
    
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

def prepare_model_parameters(dataset, var_to_predict, corr_threshold, splits_number,  var_type):
    
    best_var_correlation = get_independent_vars(dataset, var_to_predict, corr_threshold, var_type)
    
    X = dataset[best_var_correlation.keys()].values
    
    if var_type == "continuous":

        Y = dataset[var_to_predict].values
        
    elif var_type == "discrete":
        
        Y = dataset[var_to_predict+"_int"].values        
    
    kf = skms.KFold(n_splits = splits_number, shuffle = True)
    

    return best_var_correlation, X, Y, kf


def choose_continuous_var(root, dataset, continuous_vars, corr_threshold, max_regression_pow, splits_number):

    label = tk.Label(root, text="Оберіть неперервну змінну: ")

    label.pack()

    value_inside = tk.StringVar()

    cond_entry = tk.OptionMenu(root, value_inside, *continuous_vars)

    cond_entry.pack()


    def get_cont_var():

        cont_var_to_predict = value_inside.get()

        print(f"Обрано змінну {cont_var_to_predict}")
        

        best_cont_var_correlation, X, Y, kf = prepare_model_parameters(dataset, cont_var_to_predict, corr_threshold, splits_number, "continuous")

        print(f"Обрана змінна найкраще корелює зі змінними: {best_cont_var_correlation}")

        print()
        
    
        regression_powers = [i for i in range(1, max_regression_pow+1)]
        
        regression_names = {1: "Лінійна регресія", 2: "Квадратурна регресія", 3: "Кубічна регресія"}
        
        
        for rp in regression_powers:

            print(regression_names.get(rp, f"Регресія степеня {rp}"))

            bm.build_regression_model(kf, X, Y, rp)


    submit_button = tk.Button(root, text='Обрати змінну', command = lambda: get_cont_var())

    submit_button.pack()

def choose_discrete_var(root, dataset, discrete_vars, corr_threshold, splits_number, tree_parameters):

    label_1 = tk.Label(root, text="Оберіть дискретну змінну: ")

    label_1.pack()

    value_inside_1 = tk.StringVar()

    cond_entry_1 = tk.OptionMenu(root, value_inside_1, *discrete_vars)

    cond_entry_1.pack()


    def get_discr_var():

        discrete_var_to_predict = value_inside_1.get()

        print(f"Обрано змінну {discrete_var_to_predict}")
        

        best_discrete_var_correlation, X, Y, kf = prepare_model_parameters(dataset, discrete_var_to_predict, corr_threshold, splits_number, "discrete")
        
        print(f"Обрана змінна найкраще корелює зі змінними: {best_discrete_var_correlation}")

        print()



        print("Дерево прийняття рішень")
        
        bm.build_decision_tree_model(kf, X, Y, list(best_discrete_var_correlation), tree_parameters)
        
        

    submit_button = tk.Button(root, text='Обрати змінну', command = lambda: get_discr_var())
    
    submit_button.pack()
