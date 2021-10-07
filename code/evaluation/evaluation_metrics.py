#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class running different evaluation metrics to be used during cross validation 
and final test of classifier.
The metrics contain accuracy, balanced accuracy, F1 score, Cohenäs kappa, ROC. 

Created on Thu Oct 07 04:57:35 2021

@author: ptsvilodub
"""

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, cohen_kappa_score, roc_auc_score
import pandas as pd
from code.util import EVAL_METRICS

class EvaluationMetrics():
    """Computes evaluation metrics given a DataFrame with ground truth and prediction columns"""

    def __init__(self, y_true, y_pred, metrics=EVAL_METRICS):
        """Initialize evaluation given griund truth column y_true and prediction column y_pred"""
        
        self._y_true = y_true
        self._y_pred = y_pred
        self._results = pd.DataFrame(columns=list(EVAL_METRICS.keys()))
        
    def compute_metrics(self, y_true, y_pred, metrics=EVAL_METRICS):
        """
        Compute suite of evaluation metrics. 
        Arguments
        ---------
            y_true: pd.DataFrame[labels]
                Data column containing ground truth labels
            y_pred: array
                Array-like output of classifier instance containing predicted labels
            metrics: list
                list of sklearn evaluation metrics functions
                default: EVAL_METRICS
        
        Returns
        --------
            results: pd.DataFrame
                DataFrame of shape (1 x metrics) containing results (row) of all 
                evaluation metrics (columns) for given y_true and y_pred
        """
        
        # instantiate results
        results = self._results
        row_idx = len(results)
        # append metrics to results
        for label, metric in metrics.items():
            results.loc[row_idx, label] = metric(y_true, y_pred)
        
        print("    Results table: ")
        print(results)  
        # TODO: add saving df 
        return results    
