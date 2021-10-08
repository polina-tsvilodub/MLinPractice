#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 05:33:00 2021

@author: ml
"""

# Documentation of the Tweets-Going-Viral Project by group *Ultimate Power Touple 00101010*

## Evaluation Metrics

**Motivation**
We decided to implement the following metrics for our project:
* standard accuracy (as provided by the departure point code): proportion of correctly identified labels. This metric is used in many projects and can, theerfore, be interesting for comparison purposes.
* balanced accuracy (accuracy score compensated for imbalanced datasets): average of recall per class. This metric is better suited for class imbalanced datasets, whereby our dataset is an instance thereof.
* F1-score (a score combining precision and recall): it is more representative than just the accuracy due to taking into account both correct identifications as well as successes of getting the underrepresented class.
* Cohen's kappa: a more robust score for imblanaced datasets; it is also called interrater reliability score and accounts for the probability of corretc classification at random.
* ROC (Receiver Operating Characteristic (curve)): outputs AUC (area under the curve) computed on the curve resulting from plotting the true positive rate of the classifier against its false positive rate. 

As outlined above, each metric has specific advantages and disadvantages, respectively. This suite of evaluation metrics allows the user to get a comprehensive picture of the classifier performance, given diverse information. 
In general, the user is adviced to use several metrics and interpret them in combination in order to gain a deeper understanding of the quality of the classifier. 

**Implementation**
The suite of evaluation metrics is specified in `code/util.py/EVAL_METRICS` as a dictionary of the shape {"metric_name": sklearn_metric_function_name}.
The user is welcome to update this dictionary.

In `code/evaluation/` , we implement `evaluation_metrics.py` which contains the class `EvaluationMetrics`.
The class has the following method(s):
* `compute_metrics(self, y_true, y_pred, metrics=EVAL_METRIC)`: computes all evaluation metrics specified in `m̀etrics` and creates a pd.DataFrame of the results, given vectors containing the ground truth and the classifier predictions. 

## Evaluation Schema 

## Evaluation Baseline