#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 16:47:57 2021

@author: lschiesser
"""
import unittest
import pandas as pd
from code.preprocessing.stopword_remover import StopwordRemover

class TestStopword(unittest.TestCase):
    
    def setUp(self):
        self.INPUT_COLUMN = "input"
        self.OUTPUT_COLUMN = "output"
        self.strmvr = StopwordRemover(self.INPUT_COLUMN, self.OUTPUT_COLUMN)
    
    
    def test_removal_single_sentence(self):
        input_tokens = ['this', 'is', 'an', 'example', 'sentence']
        output_tokens = ['example', 'sentence']
        
        input_df = pd.DataFrame()
        input_df[self.INPUT_COLUMN] = [input_tokens]
        
        output = self.strmvr.fit_transform(input_df)
        self.assertEqual(output[self.OUTPUT_COLUMN][0], output_tokens)

if __name__ == '__main__':
    unittest.main()