import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB

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

# determine how accurate this selection of features is
def get_accuracy_of_selection(X,y):
    # create k-fold cross validation object
    kf = KFold(n_splits=10)

    # array of accuracy predictions for this selection of features
    accuracies = []

    # perform a k-fold cross validation to determine accuracy of selected features
    for train_index, test_index in kf.split(X):
        # split into testing and training data based on the splits
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # count each occurrence of the classes to determine frequency
        class_count = [0,0]
        for i in y_train:
            class_count[int(i)] += 1

        # calculate total number of observations and determine prior probability
        total = class_count[0] + class_count[1]
        prior_probability = [class_count[0]/total, class_count[1]/total]

        # define smoothing: "portion of the largest variance of all features that is added to variances for calculation stability."
        smoothing = 1e-09

        # perform a gaussian naive bayes
        gnb = GaussianNB(priors=prior_probability, var_smoothing=smoothing)
        gnb.fit(X_train, y_train)

        y_pred = gnb.predict(X_test)  # predicted class

        # y_probs = gnb.predict_proba(X_test)  # confidence in each prediction

        # for i in range(y_pred.shape[0]):
        #     if y_pred[i] != y_test[i]:
        #         print(y_probs[i])       # shows that sometimes we are really really confident in the wrong answer

        # determine how accurate we were
        size = y_test.size
        true_count = (y_test == y_pred).sum()
        accuracy_percentage = (true_count) / size

        # add to array of accuracy predictions for this selection of features
        accuracies.append(accuracy_percentage)

    # compute the mean and standard deviation of this selection of features
    mean = np.mean(accuracies)
    sd = np.std(accuracies)

    # print("MEAN: " + str(round(mean*100,2)) + "%")
    # print("STANDARD DEVIATION: " + str(round(sd*100,2)) + "%")

    return mean, sd

# determine the best set of features to predict on
def get_best_selection_of_features(X_total,y):

    # define initial maxes
    best_selection_of_features = None
    best_mean = 0
    best_sd = 0

    # brute force determine the best selection of features
    # range of for-loop is 0 or 1 to denote if the feature should be in or out
    for f0 in range(2):
        for f1 in range(2):
            for f2 in range(2):
                for f3 in range(2):
                    for f4 in range(2):
                        for f5 in range(2):
                            # identify whether or not we will be using each feature
                            chosen_features = [f0, f1, f2, f3, f4, f5]
                            tuple = ()

                            # for each feature, if we've chosen it, add the feature to our tuple
                            for feature_num in range(len(chosen_features)):
                                if bool(chosen_features[feature_num]):
                                    tuple += (X_total[:, feature_num],)

                            # so long as some features were chosen
                            if tuple != ():
                                # create a matrix of the chosen features
                                X = np.column_stack(tuple)

                                # determine the mean and standard deviation
                                mean, sd = get_accuracy_of_selection(X, y)

                                # update our best selection
                                if (mean > best_mean):
                                    best_mean = mean
                                    best_sd = sd
                                    best_selection_of_features = chosen_features

                                    # print(chosen_features)
                                    # print("MEAN: " + str(mean))
                                    # print("SD:   " + str(sd))
                                    # print("\n")

    return best_selection_of_features, best_mean, best_sd


######################################################################

# LOAD AND SPLIT DATA
filepath = "../data/hw1_trainingset.csv"

# load data and separate out feature vectors and class label
data, headers = load_data(filepath)
X, y = separate_features_from_class(data)

# normalize distribution and force positive
# X = np.absolute(normalize(X, axis=0, norm="max"))

log_base = np.e
X = np.absolute(X) ** log_base

# determine the best set of features for the predictive model
best_selection_of_features, best_mean, best_sd = get_best_selection_of_features(X,y)

# format the chosen features nicely
chosen_features = ""
for i in range(len(best_selection_of_features)):
    if bool(best_selection_of_features[i]):
        chosen_features += ("f" + str(i+1) + ", ")
chosen_features = chosen_features[:-2]

print("CHOSEN FEATURES: " + chosen_features)
print("BEST MEAN: " + str(round(best_mean*100,2)) + "%")
print("SD: " + str(round(best_sd*100,2)) + "%")