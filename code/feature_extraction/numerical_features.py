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
        return