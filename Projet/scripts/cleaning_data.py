# -*- coding: utf-8 -*-
"""Standardization and removing the outliers"""
import numpy as np
from helper_implementations import build_poly

# Separate the labels into three categories depending on the values
def split_y(y,x):
    y_0 = y[x[:, 22] == 0]
    y_1 = y[x[:, 22] == 1]
    y_23 = y[(x[:, 22] > 1)]
    return y_0, y_1, y_23

# Separate the values of the data points into three categories depending on the values
def split_x(x):
    x_0 = x[x[:, 22] == 0][:, [0, 1, 2, 3, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21]]
    x_1 = x[x[:, 22] == 1][:, [0, 1, 2, 3, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 29]]
    x_23 = x[(x[:, 22] > 1)]
    return x_0, x_1, x_23

# Return binary values -1,1 which gives the 

def label_regression(w_0,w_1,w_23,x_0,x_1,x_23,data):

    id_ = np.arange(data.shape[0])
    id0=id_[data[:,22] == 0]
    y_0=np.dot(x_0,w_0)

    id1=id_[data[:,22]==1]
    y_1=np.dot(x_1,w_1)

    id23=id_[data[:,22] > 1]
    y_23 = np.dot(x_23,w_23)

    ypred = np.concatenate((np.concatenate((y_0, y_1), axis=None),y_23),axis=None)
    id_ = np.concatenate((np.concatenate((id0, id1), axis=None),id23),axis=None)
    y = np.transpose(np.array([id_,ypred]))
    y = y[y[:,0].argsort()][:,1]
    y[np.where(y <= 0)] = -1
    y[np.where(y > 0)] = 1
    return y


#Return binary values -1,1 for the logistic regression 
def label_log(w_0,w_1,w_23,data):

    id_ = np.arange(data.shape[0])
    x_0,x_1,x_23=arrangement(data)
    id0=id_[data[:,22] == 0]
    y_0=np.dot(x_0,w_0)

    id1=id_[data[:,22]==1]
    y_1=np.dot(x_1,w_1)

    id23=id_[data[:,22] > 1]
    y_23 = np.dot(x_23,w_23)

    ypred = np.concatenate((np.concatenate((y_0, y_1), axis=None),y_23),axis=None)
    id_ = np.concatenate((np.concatenate((id0, id1), axis=None),id23),axis=None)
    y = np.transpose(np.array([id_,ypred]))
    y = y[y[:,0].argsort()][:,1]
    y[np.where(y <= 0.5)] = -1
    y[np.where(y > 0.5)] = 1
    return y

def arrangement(x):
    for i in range (0,x.shape[1]):
        
        #Replace -999 by the mean of the train set
         x[:, 0][x[:, 0] == -999.0] = np.mean(x[:, 0][x[:, 0] != -999.0])
        
    x_0,x_1,x_23=split_x(x)
    
    return x_0,x_1,x_23

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    return x, mean_x, std_x

def adapt_x_least_squares(x, deg):
    #Replace the -999 in the first column by the mean (without the -999)
    x[:, 0][x[:, 0] == -999.0] = np.mean(x[:, 0][x[:, 0] != -999.0])
    
    #Split the data w.r.t. the number of jets
    x_0, x_1, x_23 = split_x(x)
    
    #Standardization
    x_0, x_0_mean, x_0_std = standardize(x_0)
    x_1, x_1_mean, x_1_std = standardize(x_1)
    x_23, x_23_mean, x_23_std = standardize(x_23)
    
    #Polynomial expansion
    x_0=build_poly(x_0,deg)
    x_1=build_poly(x_1,deg)
    x_23=build_poly(x_23,deg)
    
    #Adding ones to the first column
    tx_0 = np.c_[np.ones((x_0.shape[0], 1)), x_0]
    tx_1 = np.c_[np.ones((x_1.shape[0], 1)), x_1]
    tx_23 = np.c_[np.ones((x_23.shape[0], 1)), x_23]
    
    return tx_0, tx_1, tx_23

def adapt_y_least_squares(y, x):
    
    #Split the data w.r.t. the number of jets 
    y_0, y_1, y_23 = split_y(y, x)
    
    return y_0, y_1, y_23

def compute_accuracy(y, y_prediction):
    ''' computes accuracy of a prediction '''
    correct = 0.0
    for i in range (len(y)):
        if(y[i] == y_prediction[i]):
            correct += 1
    accuracy = correct/len(y)
    return accuracy
