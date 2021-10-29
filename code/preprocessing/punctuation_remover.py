#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessor that removes punctuation from the original tweet text.

Created on Wed Sep 29 09:45:56 2021

@author: ptsvilodub
"""

import string
import re
from code.preprocessing.preprocessor import Preprocessor
from code.util import COLUMN_TWEET, COLUMN_PUNCTUATION 

# removes punctuation from the original tweet
# inspired by https://stackoverflow.com/a/45600350
class PunctuationRemover(Preprocessor):
    """
    Class for removing punctuation, emojis, and special characters.
    Inherits from Preprocessor class.
    """
    
    # constructor
    def __init__(self, input_col = COLUMN_TWEET, output_col = COLUMN_PUNCTUATION):
        """
        Initialize PunctuationRemover instance with default input column containing raw tweets
        and default output column for clean text.
        """
        # input column "tweet", new output column
        super().__init__([input_col], output_col)
    
    def deEmojify(text):
        """
        Custom function. Removes all Unicodes from a string
        
        Arguments
        ---------
        text: str
            Input text
        Returns
        --------
        str_de: str
            Input text stripped off elements that cannot be encoded as ascii symbols    
        """
        str_en = text.encode("ascii", "ignore")
        str_de = str_en.decode()
        return str_de
    
    # set internal variables based on input columns
    def _set_variables(self, inputs):
        """
        Sets puntuation to be removed from text.
        Arguments
        ---------
        inputs: list
            List of input columns
        """
        # store punctuation for later reference
        self._punctuation = "[{}]".format(string.punctuation)
        
    # get preprocessed column based on data frame and internal variables
    def _get_values(self, inputs):
        """
        Removes punctuation, special characters and emojis from internally saved input column 
        Arguments
        ---------
        inputs: list
            List of input columns
        Returns
        --------
        column_stripped: list
            List containing cleaned strings.
        """
        # replace punctuation with empty string
        column = inputs[0].str.replace(self._punctuation, "")
        # remove all emojis
        column_demojified = [PunctuationRemover.deEmojify(str(sent)) for sent in column]
        # replace non-alphanumeric characters
        column_stripped = [re.sub('[^A-Za-z0-9\s]+', '', str(sent)) for sent in column_demojified]
        return column_stripped
    
    