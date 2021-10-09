#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessor that removes punctuation from the original tweet text.

Created on Wed Sep 29 09:45:56 2021

@author: lbechberger
"""

import string
import re
from code.preprocessing.preprocessor import Preprocessor
from code.util import COLUMN_TWEET, COLUMN_PUNCTUATION, deEmojify

# removes punctuation from the original tweet
# inspired by https://stackoverflow.com/a/45600350
class PunctuationRemover(Preprocessor):
    
    # constructor
    def __init__(self):
        # input column "tweet", new output column
        super().__init__([COLUMN_TWEET], COLUMN_PUNCTUATION)
    
    # set internal variables based on input columns
    def _set_variables(self, inputs):
        # store punctuation for later reference
        self._punctuation = "[{}]".format(string.punctuation)
        
    # get preprocessed column based on data frame and internal variables
    def _get_values(self, inputs):
        # replace punctuation with empty string
        column = inputs[0].str.replace(self._punctuation, "")
        # remove all emojis
        column_demojified = [deEmojify(str(sent)) for sent in column]
        # replace non-alphanumeric characters
        column_stripped = [re.sub('[^A-Za-z0-9\s]+', '', str(sent)) for sent in column_demojified]
        return column_stripped
    
    