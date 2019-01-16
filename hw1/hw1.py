import pandas as pd
import numpy as np
from sklearn.naive_bayes import ComplementNB

#######################################################################################################################

# create a complement naive bayes classifier from the filename provided
def train(filename):
    # load data and separate out feature vectors and class label
    train_data = load_data(filename)
    X_train, y_train = separate_features_from_class(train_data)

    # remove unnecessary columns and add calculated columns
    X_train = shape_data(X_train)

    # count each occurrence of the classes to determine frequency
    class_count = [0, 0]
    for i in y_train:
        class_count[int(i)] += 1

    # determine prior probability of each class
    total = class_count[0] + class_count[1]
    prior_probability = [class_count[0] / total, class_count[1] / total]

    # construct classifier, fit, then return
    cnb = ComplementNB(class_prior=prior_probability)
    cnb.fit(X_train, y_train)
    return cnb

# return a prediction for the class of the given row (output is 0 or 1)
def predict(classifier, row):
    prediction = classifier.predict(row.reshape(1,-1))
    return int(prediction[0])

# general function to load and return data in the given file
def load_data(filepath):
    f = pd.read_csv(filepath, header=0)
    return f.values

# general function to split feature vectors and class labels
def separate_features_from_class(data):
    X = data[:, :-1] # feature vectors
    y = data[:, -1]  # class labels
    return X,y

# drop chosen features and add calculated column
def shape_data(data):
    # take the absolute value of all entries (used for Complement NB)
    X = np.absolute(data)

    # only include features 1, 3, 4, 5
    tuple = ()
    tuple += (X[:, 0],)
    tuple += (X[:, 2],)
    tuple += (X[:, 3],)
    tuple += (X[:, 4],)

    # add feature that sums features 3 and 4
    tuple += (X[:, 2] + X[:, 3],)

    # combine all features into np.ndarray
    return np.column_stack(tuple)

#######################################################################################################################

# define filenames
train_filename = "data/hw1_trainingset.csv"
test_filename = "data/hw1_testset.csv"

# create classifier
classifier = train(train_filename)

# prepare test data
data = load_data(test_filename)
X_test = shape_data(data)

# create array for predictions
predictions = []

# predict on each row in the test file
for row in X_test:
    predictions.append(predict(classifier, row))

# add predictions column to output.csv
outputFile = pd.read_csv(test_filename, header=0)
outputFile.insert(6,'Label',predictions)
outputFile.to_csv("data/output.csv")