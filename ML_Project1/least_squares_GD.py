from costs import *
import numpy as np

def compute_gradient(y, tx, w):
    X=tx
    e = y - np.dot(X,w)
    N= y.shape[0]
    Grad=-np.dot(e,X)/N
    return Grad  


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    w = initial_w
    for n_iter in range(max_iters):
        loss=compute_loss(y,tx,w)
        grad=compute_gradient(y,tx,w)
        w = w - gamma*grad
    return w,loss
