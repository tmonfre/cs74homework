# Thomas Monfre
# Dartmouth College CS 74, Winter 2019
# HW1: python script used to get the accuracy of a given combination of features. Uses KFold cross-validation.

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.svm import SVC

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

# determine how accurate this selection of features is
def get_accuracy_of_selection(X,y):
    # create k-fold cross validation object
    kf = KFold(n_splits=10)

    # array of accuracy predictions for this selection of features
    accuracies = []

    # confusion matrix counts
    t0_rates = []
    f0_rates = []
    t1_rates = []
    f1_rates = []

    k_count = 0

    # perform a k-fold cross validation to determine accuracy of selected features
    for train_index, test_index in kf.split(X):
        k_count += 1
        print("running split: " + str(k_count))

        # split into testing and training data based on the splits
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf = SVC(gamma=0.01, C=1, cache_size=7000, kernel="rbf")
        clf.fit(X_train, y_train)

        # get predictions
        y_pred = clf.predict(X_test)

        # confusion matrix counts
        t0 = 0  # true 0
        f0 = 0  # false 0
        t1 = 0  # true 1
        f1 = 0  # false 1

        # fill in counts
        for i in range(y_pred.shape[0]):
            # true 0
            if y_pred[i] == 0 and y_test[i] == 0:
                t0 += 1
            # false 0
            elif y_pred[i] == 0 and y_test[i] == 1:
                f0 += 1
            # true 1
            elif y_pred[i] == 1 and y_test[i] == 1:
                t1 += 1
            # false 1
            elif y_pred[i] == 1 and y_test[i] == 0:
                f1 += 1

        # compute rates
        t0_rate = t0 / (t0 + f1)
        f0_rate = f0 / (f0 + t1)
        t1_rate = t1 / (t1 + f0)
        f1_rate = f1 / (f1 + t0)

        # add to total rates over each fold
        t0_rates.append(t0_rate)
        f0_rates.append(f0_rate)
        t1_rates.append(t1_rate)
        f1_rates.append(f1_rate)

        # determine accuracy
        size = y_test.size
        true_count = (y_test == y_pred).sum()
        accuracy_percentage = (true_count) / size

        # add to array of accuracy predictions for this selection of features
        accuracies.append(accuracy_percentage)

    # compute the mean and standard deviation of this selection of features
    mean = np.mean(accuracies)
    sd = np.std(accuracies)

    # compute mean rates for confusion matrix
    mean_t0 = np.mean(t0_rates)
    mean_f0 = np.mean(f0_rates)
    mean_t1 = np.mean(t1_rates)
    mean_f1 = np.mean(f1_rates)

    return mean, sd, mean_t0, mean_f0, mean_t1, mean_f1

######################################################################

# LOAD AND SPLIT DATA
filepath = "../data/hw3_training_data.csv"

# load data and separate out feature vectors and class label
data, headers = load_data(filepath)
X, y = separate_features_from_class(data)

# choose what features to use
tuple = ()
tuple += (X[:, 3],)
tuple += (X[:, 4],)
tuple += (X[:, 5],)
X = np.column_stack(tuple)

# determine accuracy of output
mean, sd, mean_t0, mean_f0, mean_t1, mean_f1 = get_accuracy_of_selection(X,y)

print("\n")
print("MEAN ACCURACY: " + str(round(mean*100,2)) + "%")
print("SD: " + str(round(sd*100,2)))
print("\n")
print("TRUE  0: " + str(round(mean_t0*100,2)) + "%")
print("FALSE 0: " + str(round(mean_f0*100,2)) + "%")
print("TRUE  1: " + str(round(mean_t1*100,2)) + "%")
print("FALSE 1: " + str(round(mean_f1*100,2)) + "%")