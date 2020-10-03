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


def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        loss=compute_loss(y,tx,w)
        grad=compute_gradient(y,tx,w)
        w = w - gamma*grad
     
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws