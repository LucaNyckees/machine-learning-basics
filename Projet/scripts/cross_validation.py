# -*- coding: utf-8 -*-
"""cross_validation / choice of hyperparameters"""


import numpy as np
from implementations import *
import matplotlib.pyplot as plt



def cross_validation_visualization(lambds, mse_tr, mse_te,choice):
    """visualization the curves of mse_tr and mse_te."""
    if choice == 'semilogy':
        plt.semilogy(lambds, mse_tr, marker=".", color='b', label='train error')
        plt.semilogy(lambds, mse_te, marker=".", color='r', label='test error')
    elif choice == 'semilogx':
        plt.semilogx(lambds, mse_tr, marker=".", color='b', label='train error')
        plt.semilogx(lambds, mse_te, marker=".", color='r', label='test error')
    else:
        raise SyntaxWarning
        
    plt.xlabel("hyperparameter")
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



def cross_validation(y, x, k_indices, k,method, initial_w, max_iters, gamma, lambda_ ,degree):
    """Return the loss of the k th fold"""
   
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

    # Methods
    if method == 'GD':
        _,mse_te=least_squares_GD(y_te , x_te , initial_w , max_iters , gamma)
        w,mse_tr=least_squares_GD(y_tr , x_tr , initial_w , max_iters , gamma)
    elif method == 'SGD':
        _,mse_te=least_squares_SGD(y_te , x_te , initial_w , max_iters , gamma)
        w,mse_tr=least_squares_SGD(y_tr , x_tr , initial_w , max_iters , gamma)
    elif method == 'LS':
        _,mse_te=least_squares(y_te, x_te)
        w,mse_tr=least_squares(y_tr, x_tr)
    elif method == 'RR':
        _,mse_te=ridge_regression(y_te, x_te, lambda_)
        w,mse_tr=ridge_regression(y_tr, x_tr, lambda_)
    elif method == 'LR':
        _,mse_te=logistic_regression(y_te , x_te , initial_w , max_iters , gamma)
        w,mse_tr=logistic_regression(y_tr , x_tr , initial_w , max_iters , gamma)
    else:
        raise SyntaxWarning
        
    # calculate the loss for train and test data
    
    loss_te=compute_loss(y_te,x_te,w)
    loss_tr=compute_loss(y_tr,x_tr,w)

    return loss_tr, loss_te

def cross_validation_GD_SGD_LR(y, x, k_fold, max_degree,gamma, max_iters,method):
    """"Cross validation for gradient descent to find best degree and gamma"""
    if method != 'SGD' and method != 'GD' and method !='LR':
        raise SyntaxWarning
    lambda_=0
    seed = 1
    degrees = np.arange(1,max_degree+1)
    best_gamma = []
    best_rmses = []
    
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    for d in degrees:
        print('degree = ',d)
        initial_w=np.zeros(x.shape[1]*d+1)
        # define lists to store the loss of training data and test data
        rmse_tr = []
        rmse_te = []
        # cross validation
        for i in gamma:
            print('nb gamma = ',i)
            rmse_tr_tmp = []
            rmse_te_tmp = []
            for k in range(k_fold):
                loss_tr, loss_te=cross_validation(y,x,k_indices,k,method,initial_w,max_iters,i,lambda_,d)
                rmse_tr_tmp.append(loss_tr)
                rmse_te_tmp.append(loss_te)
            rmse_tr.append(np.mean(rmse_tr_tmp))
            rmse_te.append(np.mean(rmse_te_tmp))
        #grid search for the best hyperparameters
        ind_gamma_opt = np.argmin(rmse_te)
        best_gamma.append(gamma[ind_gamma_opt])
        best_rmses.append(rmse_te[ind_gamma_opt])
        
        
    ind_best_degree=np.argmin(best_rmses)
    
    return degrees[ind_best_degree], best_gamma[ind_best_degree]



def cross_validation_LS(y,x,k_fold,max_degree):
    """"Cross validation for least square with normal equations to find best degree"""
    seed = 1
    initial_w=0
    max_iters=0
    gamma=0
    lambda_=0
    degrees = np.arange(1,max_degree+1)
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    # cross validation
    for d in degrees:
        print('degree = ',d)
        rmse_tr_tmp = []
        rmse_te_tmp = []
        for k in range(k_fold):
            loss_tr, loss_te=cross_validation(y, x, k_indices, k,'LS', initial_w, max_iters, gamma, lambda_,d)
            rmse_tr_tmp.append(loss_tr)
            rmse_te_tmp.append(loss_te)
        rmse_tr.append(np.mean(rmse_tr_tmp))
        rmse_te.append(np.mean(rmse_te_tmp))
        
    #grid search for the best hyperparameters
    index_best_degree=np.argmin(rmse_te)
    
    cross_validation_visualization(degrees, rmse_tr, rmse_te,'semilogy')
    
    return degrees[index_best_degree]

def cross_validation_RR(y, x, k_fold, max_degree, lambdas):
    """"Cross validation for ridge regression to find best degree and lambda"""

    seed = 1
    initial_w=0
    max_iters=0
    gamma=0
    degrees = np.arange(1,max_degree+1)
   
    best_lambdas = []
    best_rmses = []
    
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    for d in degrees:
        print('degree = ',d)
        initial_w=np.zeros(x.shape[1]*d+1)
        # define lists to store the loss of training data and test data
        rmse_tr = []
        rmse_te = []
        # cross validation
        for i in lambdas:
            print('nb gamma = ',i)
            rmse_tr_tmp = []
            rmse_te_tmp = []
            for k in range(k_fold):
                loss_tr, loss_te=cross_validation(y,x,k_indices,k,'RR',initial_w,max_iters,gamma,i,d)
                rmse_tr_tmp.append(loss_tr)
                rmse_te_tmp.append(loss_te)
            rmse_tr.append(np.mean(rmse_tr_tmp))
            rmse_te.append(np.mean(rmse_te_tmp))
        #grid search for the best hyperparameters
        ind_lambda_opt = np.argmin(rmse_te)
        best_lambdas.append(lambdas[ind_lambda_opt])
        best_rmses.append(rmse_te[ind_lambda_opt])
        
        
    ind_best_degree=np.argmin(best_rmses)
    
    return degrees[ind_best_degree], best_lambdas[ind_best_degree]
    
   




