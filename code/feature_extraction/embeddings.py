#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 08:20:38 2021

@author: ptsvilodub
"""

from code.feature_extraction.feature_extractor import FeatureExtractor
import gensim.downloader as api
import numpy as np
from code.util import EMBEDDING_COL

class Embeddings(FeatureExtractor):
    """Class for computing embeddings tokenized as averages of word-level embeddings from tokenized sentences"""
   
    def __init__(self, input_col):
        """
        Initialize Embeddings class as subclass of FeatureExtractor.
        Arguments
        ----------
        input_col: str
            Name of input column which contains tokenized text to be embedded.
        """
        super().__init__([input_col], input_col + EMBEDDING_COL)
        
    def _set_variables(self, inputs):
        """
        Sets internal variable given input columns. Downloads pretrained GloVe embeddings, if not downloaded yet.
        Arguments
        ----------
        inputs: list(str)
            List of column names.
        """
        # import pretrained Glove word vectors pretrained on Twitter data
        self._glove_vecs = api.load('glove-twitter-25')
    
    def _get_values(self, inputs):
        """
        Compute tweet-level embeddings as average of word-level embeddings for given input column.
        Arguments
        ---------
        inputs: list(str)
            List of column names as strings
        Returns
        ---------
        sent_embeddings_arr: np.array(len(inputs[0]), 25)
            Numpy array of length of input column of 25-dimensional GloVe embeddings for each tweet/sentence.
        """
        # initialize list of sentence embeddings
        sent_embeddings = []
        # iterate over sentences
        for sent in inputs[0]:
            print(sent)
            # compute word embeddings of each sentence as a list
            # while filtering out OOV tokens
            # then compute average over all embedded words to get sentence-level embedding (25-dimensional)
            sent_embeddings.append(np.array([self._glove_vecs[word] for word in sent if self._glove_vecs.has_index_for(word)]).mean(axis=0))
            
        # transform to numpy array
        sent_embeddings_arr = np.array(sent_embeddings)
        
        return sent_embeddings_arr
    