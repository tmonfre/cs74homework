Thomas Monfre  
Dartmouth CS 74, Winter 2019  
Homework 3

# update this!!

#### Overview
I used a Complement Naive Bayes classifier for training. This required me to take the absolute value of each dataset before feeding it to the classifier.

Using cross-validation, I found features 1, 3, 4, 5 to produce the best predictions. I also found the calculated feature 3+4 (sum each value in feature 3 with its corresponding value in feature 4) increased accuracy.

In addition to the .zip file here, I have also attached my predictions in `output.csv`. You can see more about the directory structure below.

#### Directory Structure
- `/data` contains all input files and the single output file
- `/cross-validation` contains scripts for determining the best selection of features
- `/.idea` is from PyCharm


#### File Specifics

- `hw1.py` contains my implementation of `train` and `predict`. I defined a series of helper functions at the top of the script to implement these two methods. The calls to `train` and `predict` that create a classifier, predict, and add to `output.csv` is at the bottom.


- `cross-validation/determine_accuracy.py` is a script I wrote to perform a KFold cross-validation on a specific set of features to determine their accuracy. I used a confusion matrix to determine accuracy in addition to (total correct / total observations).


- `cross-validation/brute_force_all_features.py` is a script I wrote that brute-force tries all 63 possible combinations of features as well as all possible calculated features in which we sum a value from two separate features. I used it to determine what features to add and what features to drop. Because it is brute-force, it is inefficient and slow. It was used for testing and not production.

- `data/output.csv` contains my predictions. I've also submitted them separately on Canvas.
