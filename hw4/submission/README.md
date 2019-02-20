Thomas Monfre  
Dartmouth CS 74, Winter 2019  
Homework 4

#### Overview
I used an Extra Trees classifier for training. Using cross-validation, I found the most optimal combination of features to use was features 1-6 and the optimal number of estimators to use in the classifier was 250. 

In addition to the .zip file here, I have also attached my predictions in `output.csv`. You can see more about the directory structure below.

#### Directory Structure
- `/data` contains all input files and the single output file
- `/cross-validation` contains scripts for determining the best selection of features


#### File Specifics

- `hw4.py` contains my implementation of `train` and `predict`. I defined a series of helper functions at the top of the script to implement these two methods. The calls to `train` and `predict` that create a classifier, predict, and add to `output.csv` is at the bottom.


- `cross-validation/test_accuracy.py` is a script I wrote to perform a KFold cross-validation on a specific set of features and specific choice of `n_estimators` to determine their accuracy. I used a classification report to get a detailed breakdown of how accurate my classifier was on each feature.


- `cross-validation/brute_force_feature_selction.py` is a script I wrote that brute-force tries all 63 possible combinations of features. I used it to determine what features to drop. Because it uses a brute-force approach, it is inefficient and slow. It was used for testing and not production.

- `cross-validation/brute_force_feature_selction_output.txt` is a log of the console output from the above script that brute-force tries all combinations of features. I'm attaching the output as the script takes awhile to run.

- `data/output.csv` contains my predictions. I've also submitted them separately on Canvas.
