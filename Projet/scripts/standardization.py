# -*- coding: utf-8 -*-
"""Standardization and removing the outliers"""
import numpy as np

def standardize_x_tr(x):
    """Standardize and delete the outliers of the train data set """
    mean_x=np.zeros(x.shape[1])
    std_x=np.zeros(x.shape[1])
    
    for i in range (0,x.shape[1]):
        
        #Replace -999 by the mean of the features without the -999
        mean = np.mean(x[:,i][x[:,i]>-999])
        x[:,i][x[:,i]==-999] = mean
        
        #Standardization
        new_mean=np.mean(x[:,i])
        x[:,i] = x[:,i] - new_mean
        std = np.std(x[:,i])
        x[:,i] = x[:,i] / std
        
        #Saving the mean and std
        mean_x[i]=new_mean
        std_x[i]=std
        
    return x, mean_x, std_x

def standardize_x_te(x,mean_x,std_x):
    """Standardize and delete the outliers of the test data set"""
    for i in range (0,x.shape[1]):
        
        #Replace -999 by the mean of the train set
        x[:,i][x[:,i]==-999] = mean_x[i]
        
        #Standardization by the mean and std of the train test
        x[:,i] = x[:,i] - mean_x[i]
        x[:,i] = x[:,i] / std_x[i]

    return x

def arrangement(x)
    for i in range (0,x.shape[1]):
        
        #Replace -999 by the mean of the train set
        x[:,i][x[:,i]==-999] = mean_x[i]
        
    x_0,x_1,x_23=split(data)
    
    return x_0,x_1,x_23