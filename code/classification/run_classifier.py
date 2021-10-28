#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train or evaluate a single classifier with its given set of hyperparameters.

Created on Wed Sep 29 14:23:48 2021

@author: ptsvilodub
"""

import argparse, pickle
from sklearn.dummy import DummyClassifier
from sklearn.svm import LinearSVC
from code.evaluation.evaluation_metrics import EvaluationMetrics 
from code.util import EVAL_RESULTS_PATH
from sklearn.model_selection import GridSearchCV

# setting up CLI
parser = argparse.ArgumentParser(description = "Classifier")
parser.add_argument("input_file", help = "path to the input pickle file")
parser.add_argument("-s", '--seed', type = int, help = "seed for the random number generator", default = None)
parser.add_argument("-e", "--export_file", help = "export the trained classifier to the given location", default = None)
parser.add_argument("-i", "--import_file", help = "import a trained classifier from the given location", default = None)
parser.add_argument("-m", "--majority", action = "store_true", help = "majority class classifier")
parser.add_argument("-cve", "--cv_export", help = "optional path to location to store evaluation results for SVM classifier", nargs="?", default=EVAL_RESULTS_PATH + "cv_eval_results.csv")
parser.add_argument("-de", "--dummy_classifier_export", help = "optional path to location to store dummy classifier evaluation results", nargs='?', default=EVAL_RESULTS_PATH + "dummy_classifier_eval_results.csv")
parser.add_argument("-l", "--label_based", action = "store_true", help = "label frequency based classifier")
parser.add_argument("-svm", "--svm_classifier", help = "fit SVM classifier with grid search", action = "store_true")
parser.add_argument("-svmk", "--svm_k", help = "k for cross validation for fitting SVM classifier", nargs = "?", default = 5, type=int)

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
        # fit
        classifier.fit(data["features"], data["labels"])
    
    # use SVM classifier as production classifier
    if args.svm_classifier:        
        print("    production SVM classifier")
        # SVM classifier with default parameters
        classifier_svm = LinearSVC()
        # cross-validation with grid search
        # dict of hyperparameters to optimize
        parameters = {"penalty": ("l1", "l2"), 
                      "dual": [False],
                      "C": [1, 2]}
        svm_clf = GridSearchCV(
            estimator=classifier_svm, 
            param_grid=parameters, 
            refit=True, 
            cv=args.svm_k 
            )
        # call grid search and CV
        svm_clf.fit(data["features"], data["labels"])
        # fit classifier
        #classifier_svm.fit(data["features"], data["labels"])
        # get best classifier
        svm_best = svm_clf.best_estimator_
        print('     Best SVM classifier parameters: ', svm_best)
        
        # now classify the given data with best classifier instance
        prediction_svm = svm_best.predict(data["features"])

        # set up evaluator class instance for crossvalidation results of the SVM
        evaluator_cv = EvaluationMetrics(y_true=data["labels"], y_pred=prediction_svm) 
        print("     Evaluation metrics of best SVM classifier:")
        evaluator_cv.compute_metrics()
        # export crossvalidation evaluation results to default or provided location as csv
        if args.cv_export is not None:
            with open(args.cv_export, 'a') as cv_out:
                evaluator_cv._results.to_csv(cv_out)
        
        # export the trained classifier if the user wants us to do so
        if args.export_file is not None:
            with open(args.export_file, 'wb') as f_out:
                pickle.dump(svm_best, f_out)
        
# classify data with the given train data with baseline dummy classifier
prediction_dummy = classifier.predict(data["features"])     
        
# set up another evaluator instance for the dummy classifier
evaluator_dummy_classifier = EvaluationMetrics(y_true=data["labels"], y_pred=prediction_dummy)
print("    Evaluation metrics of label frequency based dummy classifier:")
evaluator_dummy_classifier.compute_metrics()
# export final classifier evaluation results to default or provided location as csv
if args.dummy_classifier_export is not None:
    with open(args.dummy_classifier_export, 'a') as cv_out:
        evaluator_dummy_classifier._results.to_csv(cv_out)
