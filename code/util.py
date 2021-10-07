#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility file for collecting frequently used constants and helper functions.

Created on Wed Sep 29 10:50:36 2021

@author: lbechberger
"""
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, cohen_kappa_score, roc_auc_score


# column names for the original data frame
COLUMN_TWEET = "tweet"
COLUMN_LIKES = "likes_count"
COLUMN_RETWEETS = "retweets_count"

# column names of novel columns for preprocessing
COLUMN_LABEL = "label"
COLUMN_PUNCTUATION = "tweet_no_punctuation"

# default column names for evaluation
COLUMN_Y_TRUE = "labels"

# default list of evaluation metrics and their names to be displayed in the output
EVAL_METRICS = {
    "accuracy": accuracy_score, 
    "balanced_accuracy": balanced_accuracy_score, 
    "F1": f1_score,
    "Cohens_kappa": cohen_kappa_score, 
    "ROC": roc_auc_score
    }
