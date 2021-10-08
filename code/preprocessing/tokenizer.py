#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 13:39:27 2021

@author: ml
"""

from nltk.tokenize import TweetTokenizer

class Tokenizer():
    
    def __init__(self):
        self.tokenized_tweets = []
        self.tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)
        
    def tokenize(self, tweets):
        for tweet in tweets:
            tokenized_tweet = self.tokenizer.tokenize(tweet)
            self.tokenized_tweets.append(tokenized_tweet)
        return self.tokenized_tweets
        