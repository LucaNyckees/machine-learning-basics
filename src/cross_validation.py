# -*- coding: utf-8 -*-
"""cross_validation / choice of hyperparameters"""


import numpy as np
from implementations import *
from helpers import *
from helper_implementations import *
from cleaning_data import *
import matplotlib.pyplot as plt


# Given from the exercies sessions, give the indices for the k_fold cross validation
def build_k_indices(y, k_fold, seed):
    """Build k indices for k-fold."""

    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


# The below function will plot a boxplot of the accuracy of the least square method with respect to the degree of the polynomial expansion
def cross_val_LS(y, x, k_fold, max_degree):
    """Cross validation for the polynomial expansion of the featres for least_squares"""

    # Initialization
    seed = 12
    degrees = np.arange(1, max_degree + 1)
    accuracies = []
    k_indices = build_k_indices(y, k_fold, seed)

    # Cross validation
    for d in degrees:
        accuracies_tmp = []
        for i in range(k_fold):
            # Spliting the data in k-1 train folds and 1 test fold
            te_indice = k_indices[i]
            tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == i)]
            tr_indice = tr_indice.reshape(-1)
            x_te = x[te_indice]
            y_te = y[te_indice]
            x_tr = x[tr_indice]
            y_tr = y[tr_indice]

            # Adapting the train data to the problem
            x_tr0, x_tr1, x_tr23 = adapt_x(x_tr, d)
            y_tr0, y_tr1, y_tr23 = adapt_y_least_squares(y_tr, x_tr)

            # Computing the weights for each part of the data
            w_0, _ = least_squares(y_tr0, x_tr0)
            w_1, _ = least_squares(y_tr1, x_tr1)
            w_23, _ = least_squares(y_tr23, x_tr23)

            # Computing the accuracy for each validation
            y_pred = label(w_0, w_1, w_23, x_tr0, x_tr1, x_tr23, x_tr, "least squares")
            accuracy = compute_accuracy(y_tr, y_pred)
            accuracies_tmp.append(accuracy)

        # Saving the accuracy for each degree
        accuracies.append(accuracies_tmp)

    # Printing the result in boxplot form
    plt.boxplot(accuracies)
    plt.title("Train error distribution")
    plt.xlabel("degree")
    plt.ylabel("accuracy")
    plt.savefig("cross_validation_LS")


# The below function will return a max_degree*len(gamma)*2 array where the element [d,i,0] is the mean of the accuracy on the k folds for degree d and gamma gamma[i] and [d,i,1] is the std of the accuracy on the k folds for degree d and gamma[i]
# This hold for the logistic regression method and the degree d is the degree of the polynomial expansion


def cross_val_Log(y, x, k_fold, max_degree, gamma):
    """Cross validation for the polynomial expansion of the featres for logistic_regression"""

    # Initialization
    seed = 1
    max_iters = 50
    degrees = np.arange(1, max_degree + 1)
    k_indices = build_k_indices(y, k_fold, seed)
    mean_std = np.zeros((max_degree, len(gamma), 2))

    # Cross validation
    for d in degrees:
        for index, j in enumerate(gamma):
            accuracies = []
            for i in range(k_fold):
                # Spliting the data in k-1 train folds and 1 test fold
                te_indice = k_indices[i]
                tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == i)]
                tr_indice = tr_indice.reshape(-1)
                x_te = x[te_indice]
                y_te = y[te_indice]
                x_tr = x[tr_indice]
                y_tr = y[tr_indice]

                # Adapting the train data to the problem
                x_tr0, x_tr1, x_tr23 = adapt_x(x_tr, d)
                y_tr0, y_tr1, y_tr23 = adapt_y_least_squares(y_tr, x_tr)

                # Computing the weights for each part of the data
                w_0, loss_tr0 = logistic_regression(
                    y_tr0, x_tr0, np.zeros(x_tr0.shape[1]), max_iters, j
                )
                w_1, loss_tr1 = logistic_regression(
                    y_tr1, x_tr1, np.zeros(x_tr1.shape[1]), max_iters, j
                )
                w_23, loss_tr23 = logistic_regression(
                    y_tr23, x_tr23, np.zeros(x_tr23.shape[1]), max_iters, j
                )

                # Computing the accuracy for each validation
                y_pred = label(
                    w_0, w_1, w_23, x_tr0, x_tr1, x_tr23, x_tr, "logistic regression"
                )
                accuracy = compute_accuracy(y_tr, y_pred)
                accuracies.append(accuracy)

            # Saving the accuracy for each degree
            mean_std[d - 1, index, 0] = np.mean(accuracies)
            mean_std[d - 1, index, 1] = np.std(accuracies)

    return mean_std
