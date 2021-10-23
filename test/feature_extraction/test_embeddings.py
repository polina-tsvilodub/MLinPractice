#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 23:58:07 2021

@author: ml
"""

import unittest
import pandas as pd
import numpy as np
import gensim.downloader as api
from code.feature_extraction.embeddings import Embeddings

class TestEmbeddings(unittest.TestCase):
    """
    Test class for tweet embedder.
    """
    
    def setUp(self):
        """
        Initilize unit test.
        """
        self.INPUT_COLUMN = "input"
        self.OUTPUT_COLUMN = "input_embedding"
        self.embdr = Embeddings(self.INPUT_COLUMN)
        self._glove_vecs = api.load('glove-twitter-25')
        
    def test_removal(self):
        """
        Define test cases for embeddings of words of different POS.
        """
        input_tokens = [['data'], ['programming'], ['computers'], ['compute'], ['efficient'], ['simple', 'sentence']]
        output_tokens = [np.array(self._glove_vecs['data']), 
                         np.array(self._glove_vecs['programming']), 
                         np.array(self._glove_vecs['computers']), 
                         np.array(self._glove_vecs['compute']),
                         np.array(self._glove_vecs['efficient']),
                         np.array([np.array(self._glove_vecs['simple']), np.array(self._glove_vecs['sentence'])]).mean(axis=0)
                         ]
        input_df = pd.DataFrame()
        input_df[self.INPUT_COLUMN] = input_tokens
        # apply PunctuationRemover to test cases        
        output = self.embdr.fit_transform(input_df)
        # compare class output to expected output
        # use workaround via np.testing to compare numpy arrays
        # use Is None because assert_array_equal returns None is arrays equal
        self.assertIsNone(np.testing.assert_array_equal(output, np.array(output_tokens)))

if __name__ == '__main__':
    unittest.main()

