#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 16:26:26 2021

@author: lschiesser
"""

import unittest
import pandas as pd
from code.preprocessing.tokenizer import Tokenizer

class TokenizerTest(unittest.TestCase):
    
    def setUp(self):
        self.INPUT_COLUMN = "input"
        self.OUTPUT_COLUMN = "output"
        self.tokenizer = Tokenizer(self.INPUT_COLUMN, self.OUTPUT_COLUMN)
    
    def test_input_columns(self):
        self.assertEqual(self.tokenizer._input_columns, [self.INPUT_COLUMN])
    
    def test_output_column(self):
        self.assertEqual(self.tokenizer._output_column, self.OUTPUT_COLUMN)
    
    def test_tokenization_single_sentence(self):
        input_text = "This is an example with an @mario mention"
        output_text = ['this', 'is', 'an', 'example', 'with', 'an', 'mention']
        
        input_df = pd.DataFrame()
        input_df[self.INPUT_COLUMN] = [input_text]
        
        tokenized = self.tokenizer.fit_transform(input_df)
        self.assertEqual(tokenized[self.OUTPUT_COLUMN][0], output_text)
        


if __name__ == '__main__':
    unittest.main()
        