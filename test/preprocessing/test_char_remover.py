#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 23:26:39 2021

@author: ptsvilodub
"""

import unittest
import pandas as pd
from code.preprocessing.punctuation_remover import PunctuationRemover

class TestChar(unittest.TestCase):
    """
    Test class for character and puntuation remover preprocessor.
    """
    
    def setUp(self):
        """
        Initilize unit test.
        """
        self.INPUT_COLUMN = "input"
        self.OUTPUT_COLUMN = "output"
        self.pncmvr = PunctuationRemover(self.INPUT_COLUMN, self.OUTPUT_COLUMN)
    
    
    def test_removal(self):
        """
        Define test cases containing punctuation, special characters, emojis.
        """
        input_tokens = ['#datascience', 'This test, sentence, with emoji 😂', 'myemail@example.biz', 'another: example,']
        output_tokens = ['datascience', 'This test sentence with emoji ', 'myemailexamplebiz', 'another example']
        input_df = pd.DataFrame()
        input_df[self.INPUT_COLUMN] = input_tokens
        # apply PunctuationRemover to test cases        
        output = self.pncmvr.fit_transform(input_df)
        # compare class output to expected output
        self.assertListEqual(list(output[self.OUTPUT_COLUMN]), output_tokens)

if __name__ == '__main__':
    unittest.main()

