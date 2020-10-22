# -*- coding: utf-8 -*-
"""cross_validation / choice of hyperparameters"""


import numpy as np
from implementations import compute_loss
from implementations import least_squares_GD
from implementations import build_poly
import matplotlib.pyplot as plt



def cross_validation_visualization(lambds, mse_tr, mse_te):
    """visualization the curves of mse_tr and mse_te."""
    plt.semilogx(lambds, mse_tr, marker=".", color='b', label='train error')
    plt.semilogx(lambds, mse_te, marker=".", color='r', label='test error')
    plt.xlabel("lambda")
    plt.ylabel("rmse")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation")

    
def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)



def cross_validation(y, x, k_indices, k, initial_w, max_iters, gamma, degree):
    """return the loss of ridge regression."""
   
    # get k'th subgroup in test, others in train
    
    te_indice = k_indices[k]
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indice = tr_indice.reshape(-1)
    x_te=x[te_indice]
    y_te=y[te_indice]
    x_tr=x[tr_indice]
    y_tr=y[tr_indice]
    
    # form data with polynomial degree
    
    x_te=build_poly(x_te,degree)
    x_tr=build_poly(x_tr,degree)

    # ridge regression
    
    _,mse_te=least_squares_GD(y_te , x_te , initial_w , max_iters , gamma)
    w,mse_tr=least_squares_GD(y_tr , x_tr , initial_w , max_iters , gamma)
    

    # calculate the loss for train and test data
    
    loss_te=compute_loss(y_te,x_te,w)
    loss_tr=compute_loss(y_tr,x_tr,w)

    return loss_tr, loss_te

def cross_validation_demo(y,x,initial_w,max_iters):
    seed = 1
    degree = 7
    k_fold = 4
    gamma = np.logspace(-4, 0, 30)
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    # cross validation
    for i in gamma:
        rmse_tr_tmp = []
        rmse_te_tmp = []
        for k in range(k_fold):
            loss_tr, loss_te=cross_validation(y,x,k_indices,k,initial_w,max_iters,i,degree)
            rmse_tr_tmp.append(loss_tr)
            rmse_te_tmp.append(loss_te)
        rmse_tr.append(np.mean(rmse_tr_tmp))
        rmse_te.append(np.mean(rmse_te_tmp))
    
    cross_validation_visualization(gamma, rmse_tr, rmse_te)
