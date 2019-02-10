# Thomas Monfre
# Dartmouth College CS 74, Winter 2019
# HW3: Submission File. Creating a Support Vector Machine from training data, then predicting on testing data.

import pandas as pd
from sklearn.svm import SVC

# create a svm classifier from the filename provided
def train(filename):
    # load data, separate out feature vectors and class label, remove unnecessary columns
    train_data = load_data(filename)
    X_train, y_train = separate_features_from_class(train_data)
    X_train = shape_data(X_train)

    print("CONSTRUCTING SVM WITH KERNEL=RBF, GAMMA=0.01, C=1")

    # construct classifier, fit, then return
    clf = SVC(gamma=0.01, C=1, cache_size=7000, kernel="rbf")
    clf.fit(X_train, y_train)
    return clf

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

# drop chosen features
def shape_data(X):
    # only include features 4,5,6
    return X[:,3:6]

#######################################################################################################################

# define filenames
train_filename = "data/hw3_training_data.csv"
test_filename = "data/hw3_test_data.csv"

# create svm
clf = train(train_filename)

# prepare test data (call to shape_data removes unnecessary columns and adds calculated column)
data = load_data(test_filename)
X_test = shape_data(data)

# create array for predictions
predictions = []

# predict on each row in the test file
for row in X_test:
    predictions.append(predict(clf, row))

# count each occurrence of the classes to determine frequency
class_count = [0,0]
for i in predictions:
    class_count[int(i)] += 1

# determine balance of class predictions
total = class_count[0] + class_count[1]

print("PREDICTED CLASS 0: " + str(round((class_count[0]/total)*100,2)) + "%")
print("PREDICTED CLASS 1: " + str(round((class_count[1]/total)*100,2)) + "%")

# add predictions column to output.csv
outputFile = pd.read_csv(test_filename, header=0)
outputFile.insert(6,'Label',predictions)
outputFile.to_csv("data/output.csv")