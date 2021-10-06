#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train or evaluate a single classifier with its given set of hyperparameters.

Created on Wed Sep 29 14:23:48 2021

@author: lbechberger
"""

import argparse, pickle
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

# setting up CLI
parser = argparse.ArgumentParser(description = "Classifier")
parser.add_argument("input_file", help = "path to the input pickle file")
parser.add_argument("-s", '--seed', type = int, help = "seed for the random number generator", default = None)
parser.add_argument("-e", "--export_file", help = "export the trained classifier to the given location", default = None)
parser.add_argument("-i", "--import_file", help = "import a trained classifier from the given location", default = None)
parser.add_argument("-l", "--label_based", action = "store_true", help = "label frequency based classifier")
parser.add_argument("-a", "--accuracy", action = "store_true", help = "evaluate using accuracy")
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

# now classify the given data
prediction = classifier.predict(data["features"])

# collect all evaluation metrics
evaluation_metrics = []
if args.accuracy:
    evaluation_metrics.append(("accuracy", accuracy_score))

# compute and print them
for metric_name, metric in evaluation_metrics:
    print("    {0}: {1}".format(metric_name, metric(data["labels"], prediction)))
    
# export the trained classifier if the user wants us to do so
if args.export_file is not None:
    with open(args.export_file, 'wb') as f_out:
        pickle.dump(classifier, f_out)