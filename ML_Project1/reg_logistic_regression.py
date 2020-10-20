import numpy as np
from logistic_regression import *

def reg_calculate_gradient(y, tx, lambda_, w):
    g=0
    X=tx
    for i in range(y.shape[0]):
        g=g-y[i]*X[:,i]+X[:,i].dot(sigmoid(X[i,:].dot(w)))
    g=g+0.5*lambda_*w
    return g


def reg_learning_by_gradient_descent(y, tx, lambda_, w, gamma):
    w_new=w-gamma*reg_calculate_gradient(y, tx, lambda_, w)
    loss=compute_loss(y,tx,w_new)
    return w,loss


def de_standardize(x, mean_x, std_x):
    """Reverse the procedure of standardization."""
    x = x * std_x
    x = x + mean_x
    return x

def reg_logistic_regression(y,tx,lambda_,initial_w,max_iters,gamma):
    w=0
    loss=0
    for i in range(max_iters):
        w,loss=reg_learning_by_gradient_descent(y, tx, lambda_, w, gamma)
    return w,loss
