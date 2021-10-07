#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class running different evaluation metrics to be used during cross validation and final test of classifier.
The metrics contain accuracy, balanced accuracy, F1 score, Cohenäs kappa, ROC 

Created on Thu Oct 07 04:57:35 2021

@author: ptsvilodub
"""

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, cohen_kappa_score, roc_auc_score
#from code.util import COLUMN_Y_TRUE

class Evaluator():
    """Computes evaluation metrics given a DataFrame with ground truth and prediction columns"""

    def __init__(self, y_true, y_pred):
        """Initialize evaluation given griund truth column y_true and prediction column y_pred"""
        
        self._y_true = y_true
        self._y_pred = y_pred
        
    def _compute_metric(self, y_true, y_pred):
        pass
    
class Metrics(Evaluator):
    """TBD"""
    
    def __init__(self, y_true, y_pred):
        super().__init__(y_true, y_pred)
        
    def _compute_metrics(self, y_true, y_pred):
        """
        Compute suite of evaluation metrics. 
        
        Returns:
            Dictionaty of shape {metric_name: metric}
        """
        
        # collect all evaluation metrics
        evaluation_metrics = {}
        # list of metrics decided upon
        metrics_list = [accuracy_score, balanced_accuracy_score, f1_score, cohen_kappa_score, roc_auc_score]
        
        # append metrics to dictionary
        for metric in metrics_list:
            evaluation_metrics[str(metric)] = metric(y_true, y_pred)
            
        return evaluation_metrics    
