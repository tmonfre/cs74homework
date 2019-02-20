# Thomas Monfre
# Dartmouth College CS 74, Winter 2019
# HW4: Submission File. Creating an ExtraTreesClassifier from multi-label training data, then predicting on testing data.

import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

# create a svm classifier from the filename provided
def train(filename):
    # load data, separate out feature vectors and class label, remove unnecessary columns
    train_data = load_data(filename)
    X_train, y_train = separate_features_from_class(train_data)

    # construct classifier, fit, then return
    classif = ExtraTreesClassifier(n_estimators=250)
    classif.fit(X_train, y_train)

    return classif

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

#######################################################################################################################

# define filenames
train_filename = "data/CS74_HW4_training_set.csv"
test_filename = "data/CS74_HW4_test_set.csv"

# create ExtraTreesClassifier and prepare test data
classif = train(train_filename)
X_test = load_data(test_filename)

# create array for predictions
predictions = []

# predict on each row in the test file
for row in X_test:
    predictions.append(predict(classif, row))

# count each occurrence of the classes to determine frequency
class_count = [0,0,0,0,0]
for i in predictions:
    class_count[int(i)-1] += 1

# determine balance of class predictions
total = 0
for i in class_count:
    total += i

for i in range(len(class_count)):
    print("PREDICTED CLASS " + str(i+1) + ": " + str(round((class_count[i] / total) * 100, 2)) + "%")

# add predictions column to output.csv
output_file = pd.read_csv(test_filename, header=0)
output_file.insert(6,'Label',predictions)
output_file.to_csv("data/output.csv")