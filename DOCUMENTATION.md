#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 05:33:00 2021

@author: ml
"""

# Documentation ofthe Tweets-Going-Viral Project by group Ultimate Power Touple 00101010

## Evaluation Metrics
We decided to implement the following metrics for our project:
* standard accuracy (as provided by the departure point code): proportion of correctly identified labels
* balanced accuracy (accuracy score compensated for imbalanced datasets): average of recall per class
* F1-score: a score combining precision and recall; it is more representative than just the accuracy due to taking into account both correct identifications as well as successes of getting the underrepresented class
* Cohen's kappa: a more robust score for imblanaced datasets; it is also called interrater reliability score and accounts for the probability of corretc classification at random
* ROC (Receiver Operating Characteristic (curve)): outputs AUC (area under the curve) computed on the curve resulting from plotting the true positive rate of the classifier against its false positive rate. 

This suite of evaluation metrics allows the user to get a comprehensive picture of the classifier performance. In general, the user is adviced to use several metrics and interpret them in combination. 

## Evaluation Schema 

## Evaluation Baseline