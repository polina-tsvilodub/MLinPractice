#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 10:36:49 2021

@author: lschiesser
"""

import numpy as np
from code.feature_extraction.feature_extractor import FeatureExtractor

class BinaryFeatureExtractor(FeatureExtractor):
    
    def __init__(self, input_columns, output_column):
        self.input_columns = input_columns
        super().__init__([input_columns], output_column)
    
    def _get_values(self, inputs):
        result = []
        # compute binary that shows whether a feature is absent (0) or present (1)
        if self.input_columns == "video":
            for item in inputs[0]:
                result.append(item)
        else:
            for item in inputs[0]:
                # the feature is present if the list is bigger than 0 and absent if it is 0
                result.append(int(len(item)>0))
        result = np.array(result)
        result = result.reshape(-1, 1)
        return result