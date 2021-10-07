#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train or evaluate a single classifier with its given set of hyperparameters.

Created on Wed Sep 29 14:23:48 2021

@author: lbechberger
"""

import argparse, pickle
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, cohen_kappa_score, roc_auc_score
from code.evaluation.evaluation_metrics import Evaluator, Metrics
from code.util import COLUMN_Y_TRUE 

# setting up CLI
parser = argparse.ArgumentParser(description = "Classifier")
parser.add_argument("input_file", help = "path to the input pickle file")
parser.add_argument("-s", '--seed', type = int, help = "seed for the random number generator", default = None)
parser.add_argument("-e", "--export_file", help = "export the trained classifier to the given location", default = None)
parser.add_argument("-i", "--import_file", help = "import a trained classifier from the given location", default = None)
parser.add_argument("-m", "--majority", action = "store_true", help = "majority class classifier")
parser.add_argument("-a", "--accuracy", action = "store_true", help = "evaluate using accuracy")
parser.add_argument("-ba", "--balanced_accuracy", action = "store_true", help = "evaluate using balanced accuracy")
parser.add_argument("-f1", "--f1_score", action = "store_true", help = "evaluate using F1 score")
parser.add_argument("-ck", "--cohens_kappa", action = "store_true", help = "evaluate using Cohen's kappa")
parser.add_argument("-roc", "--roc", action = "store_true", help = "evaluate using Receiver Operating Characteristic Curve")

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

# collect all evaluation metrics
evaluation_metrics = []
if args.accuracy:
    evaluation_metrics.append(("accuracy", accuracy_score))

# balanced accuracy
if args.balanced_accuracy:
    evaluation_metrics.append(("balanced_accuracy", balanced_accuracy_score))    
    
# F1 score
if args.f1_score:
    evaluation_metrics.append(("f1_score", f1_score))    

# Cohen's kappa score, representing inter- or intra-rater reliability
if args.cohens_kappa:
    evaluation_metrics.append(("cohens_kappa", cohen_kappa_score))

# Receiver Operating Characteristic Curve, outputs AUC value
if args.roc:
    evaluation_metrics.append(("roc", roc_auc_score))    

# compute and print them
for metric_name, metric in evaluation_metrics: 
    print("    {0}: {1}".format(metric_name, metric(data["labels"], prediction)))

print("compute eval metrics via class")
print(Metrics(data[COLUMN_Y_TRUE], prediction)._compute_metrics(data[COLUMN_Y_TRUE], prediction))    
# export the trained classifier if the user wants us to do so
if args.export_file is not None:
    with open(args.export_file, 'wb') as f_out:
        pickle.dump(classifier, f_out)