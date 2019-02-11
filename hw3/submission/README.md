Thomas Monfre  
Dartmouth CS 74, Winter 2019  
Homework 3

#### Overview
I used a Support Vector Machine classifier for training. Using cross-validation, I found the Radial Basis Function (RBF) kernel yielded the most optimal run-time and results. When I tried running a grid search on the linear and poly kernels, even after leaving my program run for 2+ hours, it had not finished computing. I therefore decided to optimize the hyper-parameters for the RBF kernel in order to get the best possible results.

Using cross-validation, I tested all possible combinations of features as well as a range of choices for `C` and `gamma` for each selection of features. Through this, I found features 4, 5, and 6 produced the most accurate predictions. I also found a `C` value of 1 and a `gamma` value of 0.01 yielded optimal results.

In addition to the .zip file here, I have also attached my predictions in `output.csv`. You can see more about the directory structure below.

#### Directory Structure
- `/data` contains all input files and the single output file
- `/cross-validation` contains scripts for determining the best selection of features


#### File Specifics

- `hw3.py` contains my implementation of `train` and `predict`. I defined a series of helper functions at the top of the script to implement these two methods. The calls to `train` and `predict` that create a classifier, predict, and add to `output.csv` is at the bottom.


- `cross-validation/test_accuracy.py` is a script I wrote to perform a KFold cross-validation on a specific set of features and specific choice of kernel and hyperparameters to determine their accuracy. I used a confusion matrix to determine accuracy in addition to (total correct / total observations).


- `cross-validation/optimize_feature_parameters.py` is a script I wrote that brute-force tries all 63 possible combinations of features as well as all possible values of `C` and `gamma` within a defined range. I used it to determine what features to add and what features to drop, as well as the optimal hyper-parameters to use for the RBF kernel. Because it uses a brute-force approach, it is inefficient and slow. It was used for testing and not production.

- `cross-validation/parameter_optimization_output.txt` is a log of the console output from the above script that brute-force tries all combinations of features and hyper-parameters. I'm attaching the output as the script takes awhile to run.

- `data/output.csv` contains my predictions. I've also submitted them separately on Canvas.
