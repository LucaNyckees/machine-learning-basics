#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 15:45:53 2020

@author: nyckeesluca
"""
from helpers import *
from costs import *

def compute_stochastic_gradient(y, tx, w):
    error = y-tx.dot(w)
    N=len(y)
    X=tx.T
    gradient = -X.dot(error)/N
    return gradient

def least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma):
    
    w = initial_w
    for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
        for n_iter in range(max_iters):
            gradient = compute_stochastic_gradient(minibatch_y, minibatch_tx, w)
            loss = compute_loss(minibatch_y, minibatch_tx, w)
            w = w-gamma*gradient
    return w,loss


