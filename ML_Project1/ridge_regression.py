import numpy as np

def ridge_regression(y, tx, lambda_):
    lambda_prime = 2*y.shape[0]*lambda_
    X = tx
    A = X.T.dot(X) + lambda_prime*np.eye(X.shape[1])
    B = A.inv()
    w = B.dot(X.T.dot(y))
    loss = y - X.dot(w)
    return w,loss

