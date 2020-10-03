#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 15:34:55 2020

@author: nyckeesluca
"""

# -*- coding: utf-8 -*-
"""Function used to compute the loss."""
import numpy as np

def compute_loss(y, tx, w):
    e = y - np.dot(tx,w)
    N=y.shape[0]
    L = (1/2*N)*np.dot(e,e)
    
    return L 
  