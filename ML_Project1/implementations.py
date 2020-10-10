#In this file, we store the 6 most important functions of the project.

import numpy as np
import matplotlib.pyplot as plt
from ridge_regression import *
from least_squares_GD import *
from least_squares_SGD import *


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    w = initial_w
    for n_iter in range(max_iters):
        loss=compute_loss(y,tx,w)
        grad=compute_gradient(y,tx,w)
        w = w - gamma*grad
    return w,loss

def least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma):
    
    w = initial_w
    for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
        for n_iter in range(max_iters):
            gradient = compute_stochastic_gradient(minibatch_y, minibatch_tx, w)
            loss = compute_loss(minibatch_y, minibatch_tx, w)
            w = w-gamma*gradient
    return w,loss

def least_squares(y, tx):
    N=y.shape[0]
    X=tx.T
    A = np.linalg.inv(X.T.dot(X))
    w = (A.dot(X)).dot(y)
    errors = (y-X.cdot(w))**2
    MSE = np.sum(errors)/N
    return w, MSE

def ridge_regression(y, tx, lambda_):
    lambda_prime = 2*y.len()*lambda_
    X = tx.T
    A = tx.dot(X) + lambda_prime*np.eyes(tx.shape[0])
    B = A.inv()
    w = B.dot(tx.dot(y))
    loss = y - X.dot(w)
    return w,loss





