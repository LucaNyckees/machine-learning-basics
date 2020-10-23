# -*- coding: utf-8 -*-
"""Standardization and removing the outliers"""
import numpy as np
from standardizarion import *


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

# Return binary values -1,1 which gives the prediction for the regression 
def label_regression(w_0,w_1,w_23,data)

    id_ = np.arange(data.shape[0])
    x_0,x_1,x_23=arrangement(data)
    id0=id_[data[:.22] == 0]
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

# Return binary values -1,1 for a data set and a weight
def prediction(w,x):
    
    ypred=np.dot(w,x)
    
    ypred[np.where(ypred <= 0.5)] = -1
    ypred[np.where(ypred > 0.5)] = 1
    return ypred


#Return binary values -1,1 for the logistic regression 
def label_log(w_0,w_1,w_23,data)

    id_ = np.arange(data.shape[0])
    x_0,x_1,x_23=arrangement(data)
    id0=id_[data[:.22] == 0]
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



    
