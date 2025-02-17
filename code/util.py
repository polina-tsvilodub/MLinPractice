#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility file for collecting frequently used constants and helper functions.

Created on Wed Sep 29 10:50:36 2021

@author: lbechberger
"""
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, cohen_kappa_score, roc_auc_score


# column names for the original data frame
COLUMN_TWEET = "tweet"
COLUMN_LIKES = "likes_count"
COLUMN_RETWEETS = "retweets_count"
COLUMN_PHOTO = "photos"
COLUMN_VIDEO = "video"
COLUMN_URL = "urls"
COLUMN_HASHTAGS = "hashtags"
COLUMN_MENTIONS = "mentions"
COLUMN_DATE = "date"
COLUMN_TIME = "time"

# column names of novel columns for preprocessing
COLUMN_LABEL = "label"
COLUMN_PUNCTUATION = "tweet_no_punctuation"
# default column for tweets without stopwords
COLUMN_STOPWORDS = "tweet_no_stopwords"
# default column of tokenized tweets
TWEET_TOKENIZED = "tweet_tokenized"
COLUMN_PHOTO_PRESENT = "photo_present"
COLUMN_VIDEO_PRESENT = "video_present"
COLUMN_URL_PRESENT = "urls_present"

# default column names for evaluation
COLUMN_Y_TRUE = "labels"

# default list of evaluation metrics and their names to be displayed in the output
EVAL_METRICS = {
    "accuracy": accuracy_score, 
    "balanced_accuracy": balanced_accuracy_score, 
    "F1": f1_score,
    "Cohens_kappa": cohen_kappa_score, 
    "ROC": roc_auc_score
    }
# default folder for storing evaluation metrics results
EVAL_RESULTS_PATH = "./results/"

# default suffix for creatnig column with stemmed output
SUFFIX_STEMMED = "_stemmed"

# default embedding input column
EMBEDDING_INPUT = "tweet_no_stopwords"
# default column suffix for embeddings column
EMBEDDING_COL = "_embedding"

# default column with hashtags
COLUMN_HASHTAG = "hashtags"