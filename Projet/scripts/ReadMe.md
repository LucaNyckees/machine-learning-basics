# Machine Learning (CS-433)
## The Higgs Boson Project

For anything to work in this project, please make sure to download the data from https://www.kaggle.com/c/higgs-boson and upload it in the main (and only) folder of the project under the name : **'train.csv'** and **'test.csv'**.

## Project structure
```
ML_Project_1
├── submit.ipynb                    #Detailed notebook on how to create a submission file and cross-validate the hyperparameters
├── clean_data.py                   #Functions used to adapt the data to the need of machine learning's method
├── cross_validation.py             #Functions used to do the cross-validation of the hyperparameters 
├── cross_validation_LS.png         #Boxplot of the cross validation for the least square method
├── helpers_implementations.py      #Functions used to help the machine learning methods
├── helpers.py                      #Functions used to load and submit the data and prediction
├── implementations.py              #Six different machine learning method
├── ReadMe.md
└── run.py                          #Script to produce a submission file with our best model
```
### Machine Learning Methods

All methods used in the project can be found in the file `implementations.py`. The input parameters of the functions can be found in the table below. They all have as output weight,loss. The loss is computed with the mean squared error. 

| Function                                                     | Details                                                      |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| least_squares_GD (y, tx, initial_w, max_iters, gamma)        | Linear regression using gradient descent                     |
| least_squares_SGD(y, tx, initial_w, max_iters, gamma)        | Linear regression using stochastic gradient descent          |
| least squares(y, tx)                                         | Least squares regression using normal equations              |
| ridge regression(y, tx, lambda )                             | Ridge regression using normal equations                      |
| logistic regression(y, tx, initial_w, max_iters, gamma)      | Logistic regression using gradient descent or SGD            |
| reg logistic regression(y, tx, lambda_ , initial_w, max iters, gamma) | Regularized logistic regression using gradient descent or SGD

Some helpers are needed to make this methods works and can be foun in the file `helpers_implementations.py`. All functions in those files are following the exercises done in CS-433 and are annoted for a clear understanding of what has been done. 

### Cleaning the data

An important part of the project comes from the pre processing of the data.\ A lot of values in the data set are -999, corresponding to meaningless or uncomputable data. The only discret feature of the data set  **'train.csv'** and **'test.csv'** is the number of jets, that can take the value 0,1,2 or 3. These two facts are linked. Indeed, when the number of jets is 0, 10 features are always equal to -999, when it is equal to 1, 8 features are always equal to -999 and there is no specification when the number of jets is equal to 2 or 3. The only feature without any linked to the number of jets is the first column. Thus we decided to split the data in 3 parts, corresponding to the number of jets (0,1 or 2 and 3). We then standardized each part of the data and used the methods on each of this part.\ Thus we also needed to adapt the prediction labels to the splitting of the data. All of this part of the project is detailled in the file `clean_data.py`.

### Cross validation

The cross validation of the hyperparameters has been explicitely coded only for the least square with normal equations and the logistic regression. However the code can be easily adapted to the others method only by changing the name of the method in `cross_validation.py`. For one parameter validation, please modify the function cross_val_LS and for two parameters validation please modify the function cross_val_Log.
The cross_val_Log takes the raw data, the maximum degree of the polynomial extension you want to test and the different gamma you want to test (as an array) as input. The output is a max_degree\*len(gamma)\*2 array where the element [d,i,0] is the mean of the accuracy on the k folds for degree d and gamma gamma[i] and [d,i,1] is the std of the accuracy on the k folds for degree d and gamma[i]. Keep in mind that this method is quite slow and that you should consider relatively small gamma (below 10^-2) to avoid computation problem with the sigmoid function.

### submit.ipynb

This notebook has been done for the reader to be able to reproduce any of the results that we have given in the project. It contains each step of our reasoning. 

### run.py

A script that reproduce the best result we have achieved. It uses the least_square method with a polynomial expansion of degree 9. You can simply run it to get a submission file called `submission.csv`. Again, please be sure to upload the data in the main (and only) folder of the project under the name **'train.csv'** and **'test.csv'** for the script to work.

## Team members
Raphaël Mirallié\
Luca Nyckees\
Clarisse Schumer

### Contact us
raphael.mirallie@epfl.ch\
luca.nyckees@epfl.ch\
clarisse.schumer@epfl.ch
