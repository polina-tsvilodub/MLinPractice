#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 10:40:57 2021

@author: lschiesser
"""

import unittest
import pandas as pd
from code.feature_extraction.binary_features import BinaryFeatureExtractor

class TestBinaryExtractor(unittest.TestCase):
    
    def setUp(self):
        self.INPUT_COLUMN = "tweet"
        self.OUTPUT_COLUMN = "output"
        self.INPUT_COLUMNS = ["photo", "video"]
        
        self.multiple_input_extractor = BinaryFeatureExtractor(self.INPUT_COLUMNS, self.OUTPUT_COLUMN)
        self.one_input_extractor = BinaryFeatureExtractor(self.INPUT_COLUMN, self.OUTPUT_COLUMN)
    
    
    def test_one_binary_feature(self):
        url_input = ["https://google.com", "https://ikw.uos.de"]
        output = 1
        
        df = pd.DataFrame()
        df[self.INPUT_COLUMN] = [url_input]
        
        df_output = self.one_input_extractor.fit_transform(df)
        self.assertEqual(df_output[0], output)
    
    def test_two_features(self):
        photo_input = ["photo 1", "photo 2"]
        video_input = 0
        output = 1
        
        df = pd.DataFrame()
        df[self.INPUT_COLUMNS[0]] = [photo_input]
        df[self.INPUT_COLUMNS[1]] = [video_input]
        
        df_output = self.multiple_input_extractor.fit_transform(df)
        self.assertEqual(df_output[0], output)

if __name__ == '__main__':
    unittest.main()