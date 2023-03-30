# -*- coding: utf-8 -*-
"""ML methods"""
import numpy as np
from helper_implementations import *


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        w = w - gamma * gradient
    loss = compute_loss(y, tx, w)
    return w, loss


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    w = initial_w
    for n_iter in range(max_iters):
        i = np.random.randint(0, tx.shape[0])
        minibatch_y = [y[i]]
        minibatch_tx = np.array([tx[i, :]])
        gradient = compute_gradient(minibatch_y, minibatch_tx, w)
        w = w - gamma * gradient
        loss = compute_loss(minibatch_y, minibatch_tx, w)
    return w, loss


def least_squares(y, tx):
    """Least square using normal equations."""
    gram = tx.T.dot(tx)
    w = np.linalg.solve(gram, tx.T.dot(y))
    loss = compute_loss(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations."""
    gram_aug = tx.T.dot(tx) + (
        np.float64(2 * tx.shape[0] * lambda_) * np.eye(tx.T.dot(tx).shape[0])
    )
    w = np.linalg.solve(gram_aug, tx.T.dot(y))
    loss = compute_loss(y, tx, w)
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression"""
    w = initial_w
    losses = []
    treshold = 1e-8
    for n_iter in range(max_iters):
        loss = calculate_loss_LR(y, tx, w)
        grad = calculate_gradient_LR(y, tx, w)
        w -= gamma * grad
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < treshold:
            break
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression"""
    w = initial_w
    losses = []
    treshold = 1e-8
    for n_iter in range(max_iters):
        loss, grad = reg_logistic_regression_help(y, tx, w, lambda_)
        w -= gamma * grad
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < treshold:
            break
    return w, loss
