# -*- coding: utf-8 -*-
"""Compute the prediction"""
from helpers import *
from implementations import *
from cleaning_data import *

#Load the train and test data
DATA_TRAIN_PATH = 'train.csv'
DATA_TEST_PATH = 'test.csv' 
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

#Adapting the train and test data 
x_0,x_1,x_23=adapt_x(tX,9)
y_0,y_1,y_23=adapt_y_least_squares(y,tX)
x_0_te,x_1_te,x_23_te=adapt_x(tX_test,9)

#Computing the weights
w_0, loss_w0 = least_squares(y_0, x_0)
w_1, loss_w1 = least_squares(y_1, x_1)
w_23, loss_w2 = least_squares(y_23, x_23)

#Labeling w.r.t the weights
y_pred_te=label(w_0,w_1,w_23,x_0_te,x_1_te,x_23_te,tX_test,'least squares')

#Creat the output file
create_csv_submission(ids_test, y_pred_te, 'submission.csv')