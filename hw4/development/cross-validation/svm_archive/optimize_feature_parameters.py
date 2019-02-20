# Thomas Monfre
# Dartmouth College CS 74, Winter 2019
# HW3: python script used to get the accuracy of a given combination of features and choice of hyperparameters

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

# determine the best choice of hyperparameters for this selection of features
def get_accuracy_of_selection(X,y):
    # define all possible C and gamma parameter values
    C_values = [0.001, 0.01, 0.1, 1, 10]
    gamma_values = [0.001, 0.01, 0.1, 1]

    # matrix of accuracy metrics for each choice of kernel
    accuracies = np.zeros((len(C_values),len(gamma_values)))
    sds = np.zeros((len(C_values),len(gamma_values)))
    mean_t0s = np.zeros((len(C_values),len(gamma_values)))
    mean_f0s = np.zeros((len(C_values),len(gamma_values)))
    mean_t1s = np.zeros((len(C_values),len(gamma_values)))
    mean_f1s = np.zeros((len(C_values),len(gamma_values)))

    # try all possible hyper-parameter values and determine which combination generates the most accurate prediction
    for i in range(len(C_values)):
        C = C_values[i]
        for j in range(len(gamma_values)):
            gamma = gamma_values[j]

            # accuracy predictions for this choice of kernel
            param_accuracies = []

            # confusion matrix counts
            t0_rates = []
            f0_rates = []
            t1_rates = []
            f1_rates = []

            # create k-fold cross validation object
            kf = KFold(n_splits=6)

            print("testing C: " + str(C))
            print("testing gamma: " + str(gamma))

            # perform a k-fold cross validation to determine accuracy of selected features
            for train_index, test_index in kf.split(X):
                # split into testing and training data based on the splits
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                svm = SVC(kernel="rbf", C=C, gamma=gamma)
                svm.fit(X_train, y_train)

                # get predictions
                y_pred = svm.predict(X_test)

                # confusion matrix counts
                t0 = 0  # true 0
                f0 = 0  # false 0
                t1 = 0  # true 1
                f1 = 0  # false 1

                # fill in counts
                for x in range(y_pred.shape[0]):
                    # true 0
                    if y_pred[x] == 0 and y_test[x] == 0:
                        t0 += 1
                    # false 0
                    elif y_pred[x] == 0 and y_test[x] == 1:
                        f0 += 1
                    # true 1
                    elif y_pred[x] == 1 and y_test[x] == 1:
                        t1 += 1
                    # false 1
                    elif y_pred[x] == 1 and y_test[x] == 0:
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
                param_accuracies.append(accuracy_percentage)

            # compute the mean and standard deviation of accuracy for this selection of features
            accuracy = np.mean(param_accuracies)
            sd = np.std(param_accuracies)

            # compute mean rates for confusion matrix
            mean_t0 = np.mean(t0_rates)
            mean_f0 = np.mean(f0_rates)
            mean_t1 = np.mean(t1_rates)
            mean_f1 = np.mean(f1_rates)

            # hold this computation of accuracy, standard deviation, and confusion matrix
            accuracies[i,j] = accuracy
            sds[i,j] = sd
            mean_t0s[i,j] = mean_t0
            mean_f0s[i,j] = mean_f0
            mean_t1s[i,j] = mean_t1
            mean_f1s[i,j] = mean_f1

    # determine best choice of gamma and C based on accuracy of output
    max_accuracy = 0
    best_i = 0
    best_j = 0

    for i in range(len(accuracies) - 1):
        for j in range(len(accuracies) - 1):
            if (accuracies[i,j] > max_accuracy):
                max_accuracy = accuracies[i,j]
                best_i = i
                best_j = j

    return accuracies[best_i,best_j], sds[best_i,best_j], C_values[best_i], gamma_values[best_j], mean_t0s[best_i,best_j], mean_f0s[best_i,best_j], mean_t1s[best_i,best_j], mean_f1s[best_i,best_j]

# determine the best set of features to predict on by trying all possible combinations of features
def get_best_selection_of_features(X_total,y):
    # define initial maxes
    best_selection_of_features = None
    best_mean = 0
    best_sd = 0
    best_confusion_matrix = []
    best_C = 0
    best_gamma = 0

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

                                mean, sd, C_choice, gamma_choice, mean_t0, mean_f0, mean_t1, mean_f1 = get_accuracy_of_selection(X,y)
                                print("\n")

                                # update our best selections
                                if (mean > best_mean):
                                    best_mean = mean
                                    best_sd = sd
                                    best_selection_of_features = chosen_features
                                    best_confusion_matrix = [mean_t0, mean_f0, mean_t1, mean_f1]
                                    best_C = C_choice
                                    best_gamma = gamma_choice

    return best_selection_of_features, best_mean, best_sd, best_C, best_gamma, best_confusion_matrix

######################################################################

# LOAD AND SPLIT DATA
filepath = "../data/CS74_HW4_training_set.csv"

# load data and separate out feature vectors and class label
data, headers = load_data(filepath)
X, y = separate_features_from_class(data)

best_selection_of_features, best_mean, best_sd, best_C, best_gamma, best_confusion_matrix = get_best_selection_of_features(X,y)

print("BEST FEATURES TO USE:")
print(best_selection_of_features)
print("BEST C: " + str(best_C))
print("BEST GAMMA: " + str(best_gamma))
print("\n")
print("MAXIMUM ACCURACY: " + str(round(best_mean*100,2)) + "%")
print("SD: " + str(round(best_sd*100,2)))
print("\n")
print("TRUE  0: " + str(round(best_confusion_matrix[0]*100,2)) + "%")
print("FALSE 0: " + str(round(best_confusion_matrix[1]*100,2)) + "%")
print("TRUE  1: " + str(round(best_confusion_matrix[2]*100,2)) + "%")
print("FALSE 1: " + str(round(best_confusion_matrix[3]*100,2)) + "%")