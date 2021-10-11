#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 12:00:44 2021

@author: lschiesser
"""

from code.preprocessing.preprocessor import Preprocessor
from code.util import TWEET_TOKENIZED, COLUMN_STOPWORDS
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')


class StopwordRemover(Preprocessor):
    
    def __init__(self):
        """
        Initiliaze class
        """
        super().__init__([TWEET_TOKENIZED], COLUMN_STOPWORDS)
    
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
        tweets_no_stopwords = []
        for tweet in inputs[0]:
            tweet_no_stopwords = [word for word in tweet if not word in stopwords.words()]
            tweets_no_stopwords.append(tweet_no_stopwords)
        
        return tweets_no_stopwords
            