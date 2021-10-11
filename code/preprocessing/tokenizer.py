#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 13:39:27 2021

@author: lschiesser
"""

from code.preprocessing.preprocessor import Preprocessor
from code.util import COLUMN_TWEET, TWEET_TOKENIZED
from nltk.tokenize import TweetTokenizer

class Tokenizer(Preprocessor):
    
    def __init__(self):
        """
        Initialize tokenizer and super class preprocessor with precoded 
        input and output column from code.util

        """
        # initialize Tokenizer
        self.tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)
        # initiailze superclass
        super().__init__([COLUMN_TWEET], TWEET_TOKENIZED)
        
    def _get_values(self, inputs):
        """
        Tokenizes the tweets given as inputs using TweetTokenizer.
        Tokenizer removes handles, does not preserve case and reduces repeating sequences to length of 3

        Parameters
        ----------
        inputs : list(string())
            List of strings containing the tweets to be tokenized.

        Returns
        -------
        tokenized_tweets : list(list(string()))
            List of lists. Each nested list represents a tokenized tweet

        """
        tokenized_tweets = []
        #iterate over each
        for tweet in inputs[0]:
            tokenized_tweet = self.tokenizer.tokenize(tweet)
            tokenized_tweets.append(tokenized_tweet)
        
        return tokenized_tweets
        