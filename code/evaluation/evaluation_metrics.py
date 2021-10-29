#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class running different evaluation metrics to be used during cross validation 
and final test of classifier.
The metrics contain accuracy, balanced accuracy, F1 score, Cohenäs kappa, ROC. 

Created on Thu Oct 07 04:57:35 2021

@author: ptsvilodub
"""

#from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, cohen_kappa_score, roc_auc_score
import pandas as pd
from code.util import EVAL_METRICS, EVAL_RESULTS_PATH
import os

class EvaluationMetrics():
    """Object containing evaluation metrics given ground truth and prediction columns"""

    def __init__(
            self, 
            y_true, 
            y_pred, 
            metrics=EVAL_METRICS, 
            eval_path=EVAL_RESULTS_PATH
            ):
        """
        Initialize evaluation given ground truth column y_true and prediction column y_pred.
        Create evaluation results directory, if not present yet.
        
        Arguments
        ----------
            y_true: pd.DataFrame[labels]
                Data column containing ground truth labels
            y_pred: array
                Array-like output of classifier instance containing predicted labels
            metrics: list
                List of sklearn evaluation metrics functions
                default: EVAL_METRICS    
            eval_path: str
                Path to directory where evaluation results will be stored
                default: EVAL_RESULTS_PATH
        """
        
        self._y_true = y_true
        self._y_pred = y_pred
        self.metrics = metrics
        self._results = pd.DataFrame(columns=list(EVAL_METRICS.keys()))
        
        # check if results directory exists, otherwise initialze
        if os.path.isdir(eval_path):
            pass
        else:
            os.mkdir(eval_path)
        
    def compute_metrics(
            self
            ):
        """
        Compute suite of evaluation metrics. 
        
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
        for label, metric in self.metrics.items():
            results.loc[row_idx, label] = metric(self._y_true, self._y_pred)
        
        print("    Results table: ")
        print(results)  
    
        return results    
