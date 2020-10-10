

import numpy as np

def ridge_regression(y, tx, lambda_):
    lambda_prime = 2*y.len()*lambda_
    X = tx.T
    A = tx.dot(X) + lambda_prime*np.eyes(tx.shape[0])
    B = A.inv()
    w = B.dot(tx.dot(y))
    loss = y - X.dot(w)
    return w,loss
    
   


#What follows is provided in lab 03, but I don't know yet if it's useful for the project.

"""def ridge_regression_demo(x, y, degree, ratio, seed):

    # define parameter
    lambdas = np.logspace(-5, 0, 15)
    # ***************************************************
    # INSERT YOUR CODE HERE
    # split the data, and return train and test data: TODO
    # ***************************************************
    raise NotImplementedError
    # ***************************************************
    # INSERT YOUR CODE HERE
    # form train and test data with polynomial basis function: TODO
    # ***************************************************
    raise NotImplementedError

    rmse_tr = []
    rmse_te = []
    for ind, lambda_ in enumerate(lambdas):
        # ***************************************************
        # INSERT YOUR CODE HERE
        # ridge regression with a given lambda
        # ***************************************************
        print("proportion={p}, degree={d}, lambda={l:.3f}, Training RMSE={tr:.3f}, Testing RMSE={te:.3f}".format(
               p=ratio, d=degree, l=lambda_, tr=rmse_tr[ind], te=rmse_te[ind]))
        
    # Plot the obtained results
    plot_train_test(rmse_tr, rmse_te, lambdas, degree)

    raise NotImplementedError
"""
