#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 10:26:08 2021

@author: lschiesser
"""

import numpy as np
from code.feature_extraction.feature_extractor import FeatureExtractor

class NumericalFeatureExtractor(FeatureExtractor):
    
    def __init__(self, input_column):
        super().__init__([input_column], "nr_{0}".format(input_column))
    
    
    def _get_values(self, inputs):
        """
        This functioni extracts the number of a feature

        Parameters
        ----------
        inputs : list(list())
            list of lists where each nested list contains hashtags or mentions. 

        Returns
        -------
        result : list(int())
            list of ints where each int represents the number of hashtags or mentions in a tweet.

        """
        result = []
        for item in inputs[0]:
            result.append(len(item))
        result = np.array(result)
        result = result.reshape(-1, 1)
        return result