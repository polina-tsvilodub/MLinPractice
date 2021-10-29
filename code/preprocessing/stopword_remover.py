#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 12:00:44 2021

@author: lschiesser
"""

from code.preprocessing.preprocessor import Preprocessor
from code.util import TWEET_TOKENIZED, COLUMN_STOPWORDS


class StopwordRemover(Preprocessor):
    
    def __init__(self, input_column = TWEET_TOKENIZED, output_column = COLUMN_STOPWORDS):
        """
        Initiliaze class
        """
        super().__init__([input_column], output_column)
    
    def _get_values(self, inputs):
        """
        

        Parameters
        ----------
        inputs : list(list(strings()))
            Tokenized tweets.

        Returns
        -------
        tweets_no_stopwords : list(list(strings()))
            Tokenized tweets without stopwords.

        """
        # for runtime reasons of the grid, we create a little cutsom stopwords list
        # based off of the nltk stopwords for English which can be found under
        # www.nltk.org/book/ch02.html
        custom_stopwords = ['the', 'a', 'an', 'and', 'in', 'of', 'to', 'is', 'are']
        tweets_no_stopwords = []
        for tweet in inputs[0]:
            tweet_no_stopwords = [word for word in tweet if not word in custom_stopwords]
            tweets_no_stopwords.append(tweet_no_stopwords)
        
        return tweets_no_stopwords
            