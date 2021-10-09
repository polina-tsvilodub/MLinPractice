#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 12:42:19 2021

@author: ptsvilodub
"""

from code.preprocessing.preprocessor import Preprocessor 
from nltk.stem.snowball import SnowballStemmer

class Stemmer(Preprocessor):
    """
    Stems the given input column. Must be tokenized already.
    """
    def __init__(self, input_col, output_col):
        """
        Initialize stemmer with given input and output column.
        """
        super().__init__([input_col], output_col)
        
    def _get_values(self, inputs):
        """
        Stem the words.
        
        Arguments
        ----------
        inputs: list(list(list()))
            List of columns containing data to be stemmed, 
            each column represenrted by a list consisting of lists of word-level tokens.
            
        Returns
        ---------
        stemmed: list(list())
            List of lists of stemmed sentences, each sentence list containing stems of the words.    
        """
        # initialize stemmer for English 
        stemmer = SnowballStemmer("english")
        
        # initilize output list to allow creating a list per column
        stemmed = []
        
        # iterate over columns
        for col in inputs[0]:
            # iterate over each sentence and append stemmed word
            stemmed.append([stemmer.stem(word) for word in col])
            
        
        return stemmed    