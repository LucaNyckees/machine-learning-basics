# -*- coding: utf-8 -*-
"""Standardization and removing the outliers"""
import numpy as np

def split_y(y,x):
    y_t0 = y[x[:, 22] == 0]
    y_t1 = y[x[:, 22] == 1]
    y_t23 = y[(x[:, 22] > 1)]
    return y_t0, y_t1, y_t23

def split_x(x):
    x_t0 = x[x[:, 22] == 0][:, [0, 1, 2, 3, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21]]
    x_t1 = x[x[:, 22] == 1][:, [0, 1, 2, 3, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 29]]
    x_t23 = x[(x[:, 22] > 1)]
    return x_t0, x_t1, x_t23