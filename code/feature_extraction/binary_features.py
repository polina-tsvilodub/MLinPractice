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
        super().__init__(input_columns, output_column)