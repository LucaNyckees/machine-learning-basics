# -*- coding: utf-8 -*-
"""cross_validation / choice of hyperparameters"""


import numpy as np
from implementations import *
from helpers import *
from helper_implementations import *
from cleaning_data import *
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

def cross_val_LS(y,x,k_fold,max_degree):
    seed = 1
    degrees = np.arange(1,max_degree+1)
    
    k_indices = build_k_indices(y,k_fold,seed)
    #Cross validation
    for d in degrees:
        print(d)
    
        for i in range(k_fold):
            te_indice = k_indices[i]
            tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == i)]
            tr_indice = tr_indice.reshape(-1)
            x_te=x[te_indice]
            y_te=y[te_indice]
            x_tr=x[tr_indice]
            y_tr=y[tr_indice]
            rmse_tr = []
            rmse_te = []
            
            x_tr0, x_tr1, x_tr23 = adapt_x(x_tr, d)
            y_tr0, y_tr1, y_tr23 = adapt_y_least_squares(y_tr, x_tr)
           
        
            w_0,_ = least_squares(y_tr0, x_tr0)
            w_1,_ = least_squares(y_tr1, x_tr1)
            w_23,_ = least_squares(y_tr23, x_tr23)
            
            y_pred = label(w_0,w_1,w_23,x_tr0,x_tr1,x_tr23,x_tr,'least squares')
            
            w = np.concatenate((np.concatenate((w_0, w_1), axis=None),w_23),axis=None)
            
            # Calculate the loss for train an test data
            print('y_te ',y_te.shape)
            print('x_te ',x_te.shape)
            print('w ',w.shape)
            
            loss_te=compute_loss(y_te,x_te,w)
            loss_tr=compute_loss(y_tr,x_tr,w)
        
            rmse_tr_tmp.append(loss_tr)
            rmse_te_tmp.append(loss_te)
        
        rmse_tr.append(np.mean(rmse_tr_tmp))
        rmse_te.append(np.mean(rmse_te_tmp))
        
    
    index_best_degree=np.argmin(rmse_te)
    
    cross_validation_visualization(degrees, rmse_tr, rmse_te,'semilogy')
    
    return degrees[index_best_degree]

def cross_val_Log(y,x,k_fold,max_degree,max_iters,gamma):
    seed = 1
    
    degrees = np.arange(1,max_degree+1)
    k_indices = build_k_indices(y,k_fold,seed)
    best_lambdas = []
    best_rmses = []
    
    for d in degrees:
        initial_w=np.zeros(x.shape[1]*d+1)
        rmse_tr = []
        rmse_te = []
        for j in gamma:
            rmse_tr_tmp = []
            rmse_te_tmp = []
            for i in range(k_fold):
                te_indice = k_indices[i]
                tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == i)]
                tr_indice = tr_indice.reshape(-1)
                x_te=x[te_indice]
                y_te=y[te_indice]
                x_tr=x[tr_indice]
                y_tr=y[tr_indice]
           
                #Split the data points
                y_tr0, y_tr1, y_tr23 = adapt_y_logistic_regression(y_tr, x_tr)
                x_tr0, x_tr1, x_tr23 = adapt_x(x_tr, d)
                
                # Do the logistic regession for each split
                w_0,_ = logistic_regression(y_tr0, x_tr0,initial_w,max_iters,gamma)
                w_1,_ = logistic_regression(y_tr1, x_tr1,initial_w,max_iters,gamma)
                w_23,_ = logistic_regression(y_tr23, x_tr23,initial_w,max_iters,gamma)
       
                y_pred = label(w_0,w_1,w_23,x_t0,x_t1,x_t23,data,'logistic regression')
                w = np.concatenate((np.concatenate((w_0, w_1), axis=None),w_23),axis=None)
                
                # Calculate the loss for train and test data
                loss_te=compute_loss(y_te,x_te,w)
                loss_tr=compute_loss(y_tr,x_tr,w)
        
                rmse_tr_tmp.append(loss_tr)
                rmse_te_tmp.append(loss_te)
        
            rmse_tr.append(np.mean(rmse_tr_tmp))
            rmse_te.append(np.mean(rmse_te_tmp))
        #Grid seach for the best hyperparameters
        ind_gamma_opt = np.argmin(rmse_te)
        best_gamma.append(gamma[ind_gamma_opt])
        best_rmses.append(rmse_te[ind_gamma_opt])
    
   
    
    ind_best_degree=np.argmin(best_rmses)
    
    return degrees[ind_best_degree], best_gamma[ind_best_degree]


    
    



