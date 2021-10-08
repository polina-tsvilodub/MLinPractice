#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train or evaluate a single classifier with its given set of hyperparameters.

Created on Wed Sep 29 14:23:48 2021

@author: ptsvilodub
"""

import argparse, pickle
from sklearn.dummy import DummyClassifier
from code.evaluation.evaluation_metrics import EvaluationMetrics 
import pandas as pd
from code.util import EVAL_RESULTS_PATH

# setting up CLI
parser = argparse.ArgumentParser(description = "Classifier")
parser.add_argument("input_file", help = "path to the input pickle file")
parser.add_argument("-s", '--seed', type = int, help = "seed for the random number generator", default = None)
parser.add_argument("-e", "--export_file", help = "export the trained classifier to the given location", default = None)
parser.add_argument("-i", "--import_file", help = "import a trained classifier from the given location", default = None)
parser.add_argument("-m", "--majority", action = "store_true", help = "majority class classifier")
parser.add_argument("-cve", "--cv_export", help = "optional path to location to store crossvalidation evaluation results", nargs='?', default=EVAL_RESULTS_PATH + "cv_eval_results.csv")
parser.add_argument("-fe", "--final_classifier_export", help = "optional path to location to store final classifier evaluation results", nargs='?', default=EVAL_RESULTS_PATH + "final_classifier_eval_results.csv")

args = parser.parse_args()

# load data
with open(args.input_file, 'rb') as f_in:
    data = pickle.load(f_in)

if args.import_file is not None:
    # import a pre-trained classifier
    with open(args.import_file, 'rb') as f_in:
        classifier = pickle.load(f_in)

else:   # manually set up a classifier
    
    if args.majority:
        # majority vote classifier
        print("    majority vote classifier")
        classifier = DummyClassifier(strategy = "most_frequent", random_state = args.seed)
        classifier.fit(data["features"], data["labels"])

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