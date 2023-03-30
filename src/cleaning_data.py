# -*- coding: utf-8 -*-
"""Cleaning the data"""
import numpy as np
from helper_implementations import build_poly


def split_y(y, x):
    """Separate the labels into three parts w.r.t the number of jet"""
    y_0 = y[x[:, 22] == 0]
    y_1 = y[x[:, 22] == 1]
    y_23 = y[(x[:, 22] > 1)]
    return y_0, y_1, y_23


def split_x(x):
    """Separate the values of the data set into three parts w.r.t the number of jet"""
    x_0 = x[x[:, 22] == 0][
        :, [0, 1, 2, 3, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    ]
    x_1 = x[x[:, 22] == 1][
        :,
        [
            0,
            1,
            2,
            3,
            7,
            8,
            9,
            10,
            11,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            23,
            24,
            25,
            29,
        ],
    ]
    x_23 = x[(x[:, 22] > 1)]
    return x_0, x_1, x_23


def label(w_0, w_1, w_23, x_0, x_1, x_23, data, choice):
    """Prediction on the data set"""

    # Initialization of the index
    id_ = np.arange(data.shape[0])

    # Prediction on the data set corresponding to 0 jet
    id0 = id_[data[:, 22] == 0]
    y_0 = np.dot(x_0, w_0)

    # Prediction on the data set corresponding to 1 jet
    id1 = id_[data[:, 22] == 1]
    y_1 = np.dot(x_1, w_1)

    # Prediction on the data set corresponding to 2 and 3 jets
    id23 = id_[data[:, 22] > 1]
    y_23 = np.dot(x_23, w_23)

    # Sticking all the prediction in one array
    ypred = np.concatenate((np.concatenate((y_0, y_1), axis=None), y_23), axis=None)

    # Ordering the array so that it corresponds the the inital data set
    id_ = np.concatenate((np.concatenate((id0, id1), axis=None), id23), axis=None)
    y = np.transpose(np.array([id_, ypred]))
    y = y[y[:, 0].argsort()][:, 1]

    # Classification w.r.t the chosen method
    if choice == "least squares":
        y[np.where(y <= 0)] = -1
        y[np.where(y > 0)] = 1
    elif choice == "logistic regression":
        y[np.where(y <= 0.5)] = -1
        y[np.where(y > 0.5)] = 1
    else:
        raise SyntaxWarning

    return y


def adapt_x(x, deg):
    """Adapt the data by spliting it and expand it with the polynomial basis"""
    # Replace the -999 in the first column by the mean (without the -999)
    x[:, 0][x[:, 0] == -999.0] = np.mean(x[:, 0][x[:, 0] != -999.0])

    # Split the data w.r.t. the number of jets
    x_0, x_1, x_23 = split_x(x)

    # Standardization
    x_0, _, _ = standardize(x_0)
    x_1, _, _ = standardize(x_1)
    x_23, _, _ = standardize(x_23)

    # Polynomial expansion
    x_0 = build_poly(x_0, deg)
    x_1 = build_poly(x_1, deg)
    x_23 = build_poly(x_23, deg)

    # Adding ones to the first column
    tx_0 = np.c_[np.ones((x_0.shape[0], 1)), x_0]
    tx_1 = np.c_[np.ones((x_1.shape[0], 1)), x_1]
    tx_23 = np.c_[np.ones((x_23.shape[0], 1)), x_23]

    return tx_0, tx_1, tx_23


def adapt_y_least_squares(y, x):
    """Adapt the labels by spliting it"""
    # Split the data w.r.t. the number of jets
    y_0, y_1, y_23 = split_y(y, x)

    return y_0, y_1, y_23


def adapt_y_logistic(y, x):
    """Adapt the labels by spliting it"""
    # For logistic regression only: bring y back to 0 and 1 -> for logistic regression #FIXME bring back
    y = np.array([0 if i == -1 else 1 for i in y])

    # Step 2: Split data set into 3 datasets depending on jet num
    y_0, y_1, y_23 = split_y(y, x)

    return y_0, y_1, y_23


def compute_accuracy(y, y_prediction):
    """Computes accuracy of a prediction"""
    right = 0.0
    for i in range(len(y)):
        if y[i] == y_prediction[i]:
            right += 1  # Each time our predictor is correct, we add 1 to the iterative variable
    accuracy = right / len(y)
    return accuracy


def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    return x, mean_x, std_x
