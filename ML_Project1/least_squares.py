#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 13:41:31 2020

@author: nyckeesluca
"""

import numpy as np
import matplotlib.pyplot as plt

# returns the optimal weights and the corresponding MSE

def least_squares(y, tx):
    N=y.shape[0]
    X=tx.T
    A = np.linalg.inv(X.T.dot(X))
    w = (A.dot(X)).dot(y)
    errors = (y-X.cdot(w))**2
    MSE = np.sum(errors)/N
    return w, MSE
    