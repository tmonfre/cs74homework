# Thomas Monfre
# Dartmouth College CS 74, Winter 2019
# HW4: python script used to brute force test the best combination of features to use

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier

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
    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # perform a random forest classification using one vs rest
    classif = ExtraTreesClassifier(n_estimators=250)
    classif.fit(X_train, y_train)

    # get predictions
    y_pred = classif.predict(X_test)

    # determine accuracy
    size = y_test.size
    true_count = (y_test == y_pred).sum()
    accuracy_percentage = (true_count) / size

    return accuracy_percentage


# determine the best set of features to predict on
def get_best_selection_of_features(X_total,y):

    # define initial maxes
    best_selection_of_features = None
    best_accuracy = 0

    # brute force determine the best selection of features
    # range of for-loop is 0 or 1 to denote if the feature should be in (1) or out (0)
    for f0 in range(2):
        for f1 in range(2):
            for f2 in range(2):
                for f3 in range(2):
                    for f4 in range(2):
                        for f5 in range(2):
                            # identify whether or not we will be using each feature
                            chosen_features = [f0, f1, f2, f3, f4, f5]

                            print("testing features:")
                            print(chosen_features)
                            print("\n")

                            # construct tuple to build table
                            tuple = ()

                            # for each feature, if we've chosen it, add the feature to our tuple
                            for feature_num in range(len(chosen_features)):
                                if bool(chosen_features[feature_num]):
                                    tuple += (X_total[:, feature_num],)

                            # so long as some features were chosen
                            if tuple != ():
                                # create a matrix of the chosen features
                                X = np.column_stack(tuple)

                                # determine the mean and standard deviation and confusion matrix
                                accuracy = get_accuracy_of_selection(X, y)

                                # update our best selections
                                if (accuracy > best_accuracy):
                                    best_accuracy = accuracy
                                    best_selection_of_features = chosen_features

    return best_selection_of_features, best_accuracy


######################################################################

# LOAD AND SPLIT DATA
filepath = "../data/CS74_HW4_training_set.csv"

# load data and separate out feature vectors and class label
data, headers = load_data(filepath)
X, y = separate_features_from_class(data)

# determine the best set of features for the predictive model
best_selection_of_features, best_accuracy = get_best_selection_of_features(X,y)

# format the chosen features nicely
chosen_features = ""
for i in range(len(best_selection_of_features)):
    if bool(best_selection_of_features[i]):
        chosen_features += ("f" + str(i+1) + ", ")
chosen_features = chosen_features[:-2]


print("CHOSEN FEATURES: " + chosen_features)
print("BEST ACCURACY: " + str(round(best_accuracy*100,2)) + "%")