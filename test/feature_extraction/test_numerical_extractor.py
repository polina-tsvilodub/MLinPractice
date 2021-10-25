#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 23:47:39 2021

@author: ptsvilodub
"""

import unittest
import pandas as pd
from code.feature_extraction.numerical_features import NumericalFeatureExtractor

class TestNumerical(unittest.TestCase):
    """
    Test class for numerical feature extractor for nr of @mentions etc.
    """
    
    def setUp(self):
        """
        Initilize unit test.
        """
        self.INPUT_COLUMN = "mentions"
        self.numft = NumericalFeatureExtractor(self.INPUT_COLUMN)
        
    def test_numerical(self):
        """
        Define test cases for lists of @ or lists of hashtags.
        """
        input_tokens = [['#data', '#computerscience'], ['@elonmusk'], [], ['@billgates', '@donaldtrump', '@stevejobs']]
        output_tokens = [2, 1, 0, 3]
        input_df = pd.DataFrame()
        input_df[self.INPUT_COLUMN] = input_tokens
        # apply PunctuationRemover to test cases        
        output = self.numft.fit_transform(input_df)
        # compare class output to expected output
        self.assertListEqual(output, output_tokens)

if __name__ == '__main__':
    unittest.main()

