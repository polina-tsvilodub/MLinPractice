#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 12:42:19 2021

@author: ptsvilodub
"""

from code.preprocessing.preprocessor import Preprocessor 
import nltk

class Stemmer(Preprocessor):
    """
    Stems the given input column. (must be tolenized already?)
    """
    def __init__(self, [input_col], output_col):
        super().__init__([input_col], output_col)
        
    def _get_values(self, inputs):
        """
        Stem the words.
        """
        #TODO
        pass