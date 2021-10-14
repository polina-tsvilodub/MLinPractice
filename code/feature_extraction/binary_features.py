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
        super().__init__(input_columns, output_column)
    
    def _get_values(self, inputs):
        result = []
        # only process feature if it's a list
        if type(inputs[0][0]) == "list":
            # one feature is combined from two columns in the dataset called media present
            # it is combined from the columns photos and video
            if "photos" in self.input_columns:
                # column photos comes as a list of json objects
                # column photos comes as a boolean number indicating presence (1) or absence (0) of video
                for tmp in inputs[0]:
                    # this expression returns 1 if there is media persent (if list of photos bigger than 0
                    # and/or if there video == 1) and 0 if media is absent (list of photos == 0 and video == 0)
                    result.append(int(len(tmp[0]) > 0 or bool(tmp[1])))
            else:
                # this feature is not a combined feature, it rather showss the absence (0) or presence (1) of it
                for link in inputs[0]:
                    # the feature is present if the list is bigger than 0 and absent if it is 0
                    result.append(int(len(link)>0))
        else:
            return
        result = np.array(result)
        return result