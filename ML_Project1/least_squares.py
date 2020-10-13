import numpy as np
import matplotlib.pyplot as plt

# returns the optimal weights and the corresponding loss

def least_squares(y, tx):
    N=y.shape[0]
    X=tx
    A = np.linalg.inv(X.T.dot(X))
    w = (A.dot(X.T)).dot(y)
    errors = (y-X.dot(w))**2
    MSE = np.sum(errors)/N
    return w, MSE
    
