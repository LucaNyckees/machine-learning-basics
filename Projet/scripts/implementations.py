# -*- coding: utf-8 -*-
"""ML methods"""
import numpy as np
from helper_implementations import *

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
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, 1):
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

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    losses=[]
    treshold=1e-8
    for n_iter in range (max_iters):
        loss=calculate_loss_LR(y,tx,w)
        grad=calculate_gradient_LR(y,tx,w)
        w -= gamma*grad
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return w,loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    w = initial_w
    losses=[]
    treshold=1e-8
    for n_iter in range (max_iters):
        loss,grad=reg_logistic_regression_help(y,tx,w,lambda_)
        w -= gamma*grad
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return w,loss

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