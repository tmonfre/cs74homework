# Thomas Monfre
# Dartmouth College CS 74, Winter 2019
# HW1: python script used to get the accuracy of a given combination of features. Uses KFold cross-validation.

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# load the data from the given filepath
def load_data(filepath):
    f = pd.read_csv(filepath, header=0)  # read input file
    headers = list(f.columns.values)     # put the original column names in a python list
    array = f.values                     # create a numpy array for input into scikit-learn
    return array, headers

# split feature vectors and class labels
def separate_features_from_class(data):
    X = data[:, :6] # feature vectors (remove feature 1)
    y = data[:, 6]  # class labels
    return X,y

def perform_grid_search(X,y,folds):
    C = [0.001, 0.01, 0.1, 1, 10]
    gamma = [0.001, 0.01, 0.1, 1]

    grid_parameters = {
        "C": C,
        "gamma": gamma
    }

    grid_search = GridSearchCV(SVC(kernel='rbf'), grid_parameters, cv=folds)
    grid_search.fit(X, y)

    return grid_search.best_params_

######################################################################

# LOAD AND SPLIT DATA
filepath = "../data/hw3_training_data.csv"

# load data and separate out feature vectors and class label
data, headers = load_data(filepath)
X, y = separate_features_from_class(data)

# only include features 4,5,6
tuple = ()
tuple += (X[:, 3],)
tuple += (X[:, 4],)
tuple += (X[:, 5],)

# combine all features into np.ndarray
X = np.column_stack(tuple)

# perform grid search to get best C and gamma parameter values to use
print("BEST PARAMETERS TO USE:")
print(perform_grid_search(X,y,6))