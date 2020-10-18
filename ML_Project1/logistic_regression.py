import numpy as np
from costs import *


def sigmoid(t):
    return (np.exp(t))/(1+np.exp(t))


def calculate_loss(y, tx, w):
    L=0
    X=tx
    for i in range(y.shape[0]):
        L=L-y[i]*X[i,:].dot(w)+np.log(1+np.exp(X[i,:].dot(w)))
    return -L


def calculate_gradient(y, tx, w):
    g=0
    X=tx
    for i in range(y.shape[0]):
        g=g-y[i]*X[:,i]+X[:,i].dot(sigmoid(X[i,:].dot(w)))
    return g


def learning_by_gradient_descent(y, tx, w, gamma):
    w_new=w-gamma*calculate_gradient(y, tx, w)
    loss=compute_loss(y,tx,w_new)
    return w,loss


def de_standardize(x, mean_x, std_x):
    """Reverse the procedure of standardization."""
    x = x * std_x
    x = x + mean_x
    return x

def logistic_regression(y,tx,initial_w,max_iters,gamma):
    w=0
    loss=0
    for i in range(max_iters):
        w,loss=learning_by_gradient_descent(y, tx, w, gamma)[0]
    return w,loss



