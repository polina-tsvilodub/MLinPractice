#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train or evaluate a single classifier with its given set of hyperparameters.

Created on Wed Sep 29 14:23:48 2021

@author: ptsvilodub
"""

import argparse, pickle
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from code.evaluation.evaluation_metrics import EvaluationMetrics 
from code.util import EVAL_RESULTS_PATH
from sklearn.model_selection import StratifiedKFold, GridSearchCV

# setting up CLI
parser = argparse.ArgumentParser(description = "Classifier")
parser.add_argument("input_file", help = "path to the input pickle file")
parser.add_argument("-s", '--seed', type = int, help = "seed for the random number generator", default = None)
parser.add_argument("-e", "--export_file", help = "export the trained classifier to the given location", default = None)
parser.add_argument("-i", "--import_file", help = "import a trained classifier from the given location", default = None)
parser.add_argument("-m", "--majority", action = "store_true", help = "majority class classifier")
parser.add_argument("-cve", "--cv_export", help = "optional path to location to store crossvalidation evaluation results", nargs="?", default=EVAL_RESULTS_PATH + "cv_eval_results.csv")
parser.add_argument("-fe", "--final_classifier_export", help = "optional path to location to store final classifier evaluation results", nargs='?', default=EVAL_RESULTS_PATH + "final_classifier_eval_results.csv")
parser.add_argument("-l", "--label_based", action = "store_true", help = "label frequency based classifier")
paraser.add_argument("-svm", "--svm", help = "fit SVM classifier using k-fold cross-validation with grid search", nargs = "?", default = 5)

args = parser.parse_args()

# load data
with open(args.input_file, 'rb') as f_in:
    data = pickle.load(f_in)

if args.import_file is not None:
    # import a pre-trained classifier
    with open(args.import_file, 'rb') as f_in:
        classifier = pickle.load(f_in)

else:   # manually set up a classifier
    
    if args.label_based:
        print("    label frequency based dummy classifier")
        # label frequency based dummy classifier
        classifier = DummyClassifier(strategy = "stratified", random_state = args.seed)
        # use Stratified 5 fold cross validation to train classifier
        skf = StratifiedKFold(n_splits=5, shuffle=False)
        for train_index, test_index in skf.split(data):
            # split data according to indices
            X_train = data.iloc[train_index].loc[:, "features"]
            X_test = data.iloc[test_index].loc[:, "features"]
            y_train = data.iloc[train_index].loc[:, "labels"]
            y_test = data.iloc[test_index].loc[:, "labels"]
            
            classifier.fit(X_train, y_train)
    # use SVM classifier bz default
    if args.svm:        
        print("    production SVM classifier")
        # SVM classifier with default parameters
        classifier = SVC()
        # fit
        classifier.fit(data["features"], data["labels"])
        # cross-validation TBD
        
# now classify the given data
prediction = classifier.predict(data["features"])

# set up evaluator class instance for crossvalidation
evaluator_cv = EvaluationMetrics(y_true=data["labels"], y_pred=prediction) 
evaluator_cv.compute_metrics()
# export crossvalidation evaluation results to default or provided location as csv
if args.cv_export is not None:
    with open(args.cv_export, 'a') as cv_out:
        evaluator_cv._results.to_csv(cv_out)
        
# TODO: get best classifier instance after cross validation

# set up another evaluator instance for the final classifier
# adjust y_pred
evaluator_final_classifier = EvaluationMetrics(y_true=data["labels"], y_pred=prediction)
evaluator_final_classifier.compute_metrics()
# export final classifier evaluation results to default or provided location as csv
if args.final_classifier_export is not None:
    with open(args.final_classifier_export, 'a') as cv_out:
        evaluator_final_classifier._results.to_csv(cv_out)

# export the trained classifier if the user wants us to do so
if args.export_file is not None:
    with open(args.export_file, 'wb') as f_out:
        pickle.dump(classifier, f_out)