# -*- coding: utf-8 -*-
"""some helper functions implementations"""
import numpy as np

def compute_loss(y, tx, w):
    """Compute the MSE"""
    e=y-tx.dot(w)
    return e.dot(e)/(2*len(e))

def compute_loss_SGD(y, tx, w):
    """Compute the MSE"""
    e=y-tx.dot(w)
    return e*e/2

def compute_gradient_SGD(y, tx, w):
    """Compute the gradient"""
    error = y-tx.dot(w)
    N=len(y)
    gradient = -tx.T.dot(error)/N
    return gradient

def compute_gradient_SGD(y, tx, w):
    """Compute the gradient"""
    e = y - np.dot(tx,w)
    return -e*tx

def build_poly(x, degree):
    """Polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly=np.ones(len(x))
    for deg in range(1,degree+1):
        poly = np.c_[poly,np.power(x,deg)]
    return poly

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]           
            
def reg_logistic_regression_help(y, tx, w, lambda_):
    num_samples=y.shape[0]
    loss=calculate_loss_LR(y,tx,w)+lambda_*squeeze(w.T.dit(w))
    gradient=calculate_gradient_LR(y,tx,w)+2*lambda_*w
    return loss,gradient

def sigmoid(t):
    return 1.0/(1+np.exp(-t))

def calculate_loss_LR(y, tx, w):
    pred=sigmoid(tx.dot(w))
    loss=y.T.dot(np.log(pred)) + (1-y).T.dot(np.log(1-pred))
    return np.squeeze(- loss)

def calculate_gradient_LR(y,tx,w):
    pred=sigmoid(tx.dot(w))
    grad=tx.T.dot(pred-y)
    return grad
