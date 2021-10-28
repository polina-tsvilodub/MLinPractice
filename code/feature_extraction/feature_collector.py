#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collects the feature values from many different feature extractors.

Created on Wed Sep 29 12:36:01 2021

@author: lbechberger
"""

import numpy as np
from code.feature_extraction.feature_extractor import FeatureExtractor

# extend FeatureExtractor for the sake of simplicity
class FeatureCollector(FeatureExtractor):
    
    # constructor
    def __init__(self, features):
        
        # store features
        self._features = features
        
        # collect input columns
        # some extractors use multiple columns therefore we may have to flatten the output of get_input_columns()
        input_columns = []
        for feature in self._features:
            tmp = feature.get_input_columns()
            # if first entry of tmp (which are the returned input columns) is a list flatten it by append the columns one by one to input_columns
            if type(tmp[0]) == list:
                for item in tmp[0]:
                    input_columns.append(item)
            # if first entry is not a list (therefore only a string) append it to input_columns
            else:
                input_columns += tmp
                
        # remove duplicate columns
        input_colums = list(set(input_columns))
        
        # call constructor of super class
        super().__init__(input_columns, "FeatureCollector")

    
    # overwrite fit: instead of calling _set_variables(), we forward the call to the features
    def fit(self, df):
        
        for feature in self._features:
            feature.fit(df)

    # overwrite transform: instead of calling _get_values(), we forward the call to the features
    def transform(self, df):
        
        all_feature_values = []
        
        for feature in self._features:
            tmp = feature.transform(df)
            print(feature.get_input_columns(), tmp.shape)
            all_feature_values.append(tmp)
        
        result = np.concatenate(all_feature_values, axis = 1)
        return result

    def get_feature_names(self):
        feature_names = []
        for feature in self._features:
            feature_names.append(feature.get_feature_name())
        return feature_names