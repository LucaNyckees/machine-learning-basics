# -*- coding: utf-8 -*-
"""Some helper functions for implementations"""
import numpy as np


def compute_loss(y, tx, w):
    """Compute the MSE"""
    e = y - tx.dot(w)
    return e.T.dot(e) / (2 * len(e))


def compute_gradient(y, tx, w):
    """Compute the gradient"""
    error = y - tx.dot(w)
    N = len(y)
    gradient = -tx.T.dot(error) / N
    return gradient


def build_poly(x, degree):
    """Polynomial basis functions for input data x"""
    poly = x
    for deg in range(2, degree + 1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly


def reg_logistic_regression_help(y, tx, w, lambda_):
    """Compute loss and gradient for logistic regression"""
    num_samples = y.shape[0]
    loss = calculate_loss_LR(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
    gradient = calculate_gradient_LR(y, tx, w) + 2 * lambda_ * w
    return loss, gradient


def sigmoid(t):
    """Sigmoid function"""
    return 1.0 / (1 + np.exp(-t))


def calculate_loss_LR(y, tx, w):
    """Compute loss for logistic regression"""
    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    return np.squeeze(-loss)


def calculate_gradient_LR(y, tx, w):
    """Compute gradient for logistic regression"""
    pred = sigmoid(tx.dot(w))
    grad = tx.T.dot(pred - y)
    return grad
