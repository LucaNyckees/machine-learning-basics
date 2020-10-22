# -*- coding: utf-8 -*-
"""ML methods"""
import numpy as np

def compute_loss(y, tx, w):
    """Compute the MSE"""
    e=y-tx.dot(w)
    return e.dot(e)/(2*len(e))

def compute_gradient(y, tx, w):
    e = y - np.dot(tx,w)
    N= y.shape[0]
    return -np.dot(e,tx)/N

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    a=np.repeat(x[:, np.newaxis],degree+1,axis=1) #POUR METTRE ARRAY EN COLONE x[:,np.newaxis]
    b=np.arange(degree+1)
    return np.power(a,b)


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


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    w = initial_w
    for n_iter in range(max_iters):
        gradient=compute_gradient(y,tx,w)
        w=w-gamma*gradient
    loss=compute_loss(y,tx,w)
    return w,loss

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    w = initial_w
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 1):
        for n_iter in range(max_iters):
            gradient = compute_gradient(minibatch_y, minibatch_tx, w)
            w = w-gamma*gradient
        loss = compute_loss(minibatch_y, minibatch_tx, w)
    return w,loss

def least_squares(y, tx):
    """Least square using normal equations."""
    gram=tx.T.dot(tx)
    w=np.linalg.solve(gram,tx.T.dot(y))
    loss = compute_loss(y,tx,w)
    return w,loss

def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations."""
    gram_aug=tx.T.dot(tx)+(np.float64(2*tx.shape[0]*lambda_)*np.eye(tx.T.dot(tx).shape[0]))
    w=np.linalg.solve(gram_aug,tx.T.dot(y))
    loss = compute_loss(y,tx,w)
    return w,loss