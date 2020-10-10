#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 15:38:48 2020

@author: nyckeesluca
"""

# -*- coding: utf-8 -*-
"""Gradient Descent"""

from costs import *
import numpy as np

def compute_gradient(y, tx, w):
    e = y - np.dot(tx,w)
    N= y.shape[0]
    Grad=-np.dot(e,tx)/N
    return Grad  


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    w = initial_w
    for n_iter in range(max_iters):
        loss=compute_loss(y,tx,w)
        grad=compute_gradient(y,tx,w)
        w = w - gamma*grad
    return w,loss