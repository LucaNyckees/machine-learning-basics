#This is our main, where we make use of our functions on the given data.

import numpy as np
import matplotlib.pyplot as plots
import numpy.matlib
import datetime 
from gradient_descent import *
from stochastic_gradient_descent import *

from proj1_helpers import *
DATA_TRAIN_PATH = 'train.csv' # TODO: download train data and supply path here 
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

DATA_TEST_PATH = 'train.csv' # TODO: download train data and supply path here 
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

OUTPUT_PATH = 'test.csv' # TODO: fill in desired name of output file for submission
weights=np.zeros(30)
y_pred = predict_labels(weights, tX_test)
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)

y,tx=build_model_data(y_pred,tX_test)

"""print(tx)
print(y)
"""



from implementations import *

# Define the parameters of the algorithm.
max_iters = 50
gamma = 0.7
batch_size = 1

# Initialization
w_initial = np.zeros(31)

w_GD=least_squares_GD(y, tx, w_initial, max_iters, gamma)[0]
loss_GD=least_squares_GD(y, tx, w_initial, max_iters, gamma)[1]

print('w_GD=',w_GD,'\n','loss_GD=',loss_GD)

w_SGD=least_squares_SGD(y, tx, w_initial, max_iters, batch_size, gamma)[0]
loss_SGD=least_squares_SGD(y, tx, w_initial, max_iters, batch_size, gamma)[1]

print('w_SGD=',w_SGD,'\n','loss_SGD=',loss_SGD)
