# Thomas Monfre
# Dartmouth College CS 74, Winter 2019
# HW4: python script used to get the accuracy of a given combination of features

import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report

# load the data from the given filepath
def load_data(filepath):
    f = pd.read_csv(filepath, header=0)  # read input file
    headers = list(f.columns.values)     # put the original column names in a python list
    array = f.values                     # create a numpy array for input into scikit-learn
    return array, headers

# split feature vectors and class labels
def separate_features_from_class(data):
    X = data[:, :6] # feature vectors
    y = data[:, 6]  # class labels
    return X,y

######################################################################

# LOAD AND SPLIT DATA
filepath = "data/CS74_HW4_training_set.csv"

# load data and separate out feature vectors and class label
data, headers = load_data(filepath)
X, y = separate_features_from_class(data)

classes = ["Class 1", "Class 2", "Class 3", "Class 4", "Class 5"]

kf = KFold(n_splits=10)
accuracies = []

# perform a k-fold cross validation to determine accuracy of selected features
for train_index, test_index in kf.split(X):
    # split into testing and training data based on the splits
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


    classif = ExtraTreesClassifier(n_estimators=250)
    classif.fit(X_train, y_train)

    pred = classif.predict(X_test)

    correct_prediction_count = 0

    for i in range(len(pred)):
        if pred[i] == y_test[i]:
            correct_prediction_count += 1

    print(classification_report(y_test, pred, target_names=classes))

    accuracies.append(correct_prediction_count / len(pred))

print(np.mean(accuracies))