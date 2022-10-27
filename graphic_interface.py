# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 12:07:05 2022

@author: romas
"""

import tkinter as tk

import sklearn.model_selection as skms

import build_model as bm


def main_work(dataset, continuous_vars, discrete_vars, corr_threshold, splits_number):

    root = tk.Tk()

    root.title("Вибір змінних")

    root.geometry('700x500')

    choose_continuous_var(root, dataset, continuous_vars, corr_threshold, splits_number)

    choose_discrete_var(root, discrete_vars)

    root.mainloop()



def choose_continuous_var(root, dataset, continuous_vars, corr_threshold, splits_number):

    label = tk.Label(root, text="Оберіть неперервну змінну: ")

    label.pack()

    value_inside = tk.StringVar()

    cond_entry = tk.OptionMenu(root, value_inside, *continuous_vars)

    cond_entry.pack()


    def get_cont_var():

        var_to_predict = value_inside.get()

        print(f"Обрано змінну {var_to_predict}")


        corr = dataset.corr()

        var_correlation = dict(corr[var_to_predict])

        best_var_correlation = {k: v for k, v in var_correlation.items() if abs(v) < corr_threshold[1] and abs(v) > corr_threshold[0]}

        print(f"Обрана змінна найкраще корелює зі змінними: {best_var_correlation}")

        print()



        X = dataset[best_var_correlation.keys()].values

        Y = dataset[var_to_predict].values


        kf = skms.KFold(n_splits = splits_number, shuffle = True)


        print("Лінійна регресія")

        bm.build_regression_model(kf, X, Y, 1)

        print("Квадратурна регресія")

        bm.build_regression_model(kf, X, Y, 2)




    submit_button = tk.Button(root, text='Обрати змінну', command = lambda: get_cont_var())

    submit_button.pack()

def choose_discrete_var(root, discrete_vars):

    label_1 = tk.Label(root, text="Оберіть дискретну змінну: ")

    label_1.pack()

    value_inside_1 = tk.StringVar()

    cond_entry_1 = tk.OptionMenu(root, value_inside_1, *discrete_vars)

    cond_entry_1.pack()


    def get_discr_var():

        binary_var_to_predict = value_inside_1.get()

        print(f"Обрано змінну {binary_var_to_predict}")


        # Як обирати змінні для X?


        # corr = dataset.corr()

        # binary_var_correlation = dict(corr[binary_var_to_predict])

        # best_binary_var_correlation = {k: v for k, v in binary_var_correlation.items() if abs(v) < corr_threshold[1] and abs(v) > corr_threshold[0]}

        # print(best_binary_var_correlation)

        # print()



        # X = dataset[best_binary_var_correlation.keys()].values

        # # Y = dataset[var_to_predict].values

        # dataset[binary_var_to_predict] = dataset[binary_var_to_predict] == "Yes"

        # Y1 = dataset[binary_var_to_predict].values


        # kf = skms.KFold(n_splits = splits_number, shuffle = True)


        # print("Дерево прийняття рішень")

        # bm.build_decision_tree_model(kf, X, Y1)

    submit_button = tk.Button(root, text='Обрати змінну', command = lambda: get_discr_var())
    submit_button.pack()
