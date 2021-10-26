#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 10:30:20 2021

@author: lschiesser
"""

from code.feature_extraction.feature_extractor import FeatureExtractor
import datetime
import numpy as np

class DateExtractor(FeatureExtractor):
    
    def __init__(self, input_column, output_columns):
        super().__init__([input_column], output_columns)
    
    def _get_values(self, inputs):
        """
        Extracts month and day of month as integers from date column to make it suitable for machine learning algorithm
        
        Parameters
        ----------
        inputs : list(string())
            Date column which contains the date the tweet was published as a string.

        Returns
        -------
        results : list(list(int()))
            list of lists which contains the month and the day of month (both as int) when the tweet was published.

        """
        results = []
        for date_string in inputs[0]:
            # transform date string into datetime object to extract information more easily
            date = datetime.datetime.strptime(date_string, "%Y-%m-%d")
            # extract month and day (of month) as ints
            month = date.month
            day = date.day
            results.append([month, day])
        results = np.array(results)
        return results

class TimeExtractor(FeatureExtractor):
    
    def __init__(self, input_column, output_columns):
        super().__init__([input_column], output_columns)
        
    def _get_values(self, inputs):
        """
        Extracts hour and minutes as integers from time column to make it suitable for machine learning algorithm

        Parameters
        ----------
        inputs : list(string())
            Time column which contains the time the tweet was published as a string.

        Returns
        -------
        results : list(list(int()))
            list of lists which contains the hour and minutes (both as int) of the tweet's publication.

        """
        results = []
        for time_string in inputs[0]:
            time = datetime.datetime.strptime(time_string, "%H:%M:%S")
            hour = time.hour
            minutes = time.minute
            results.append([hour, minutes])
        results = np.array(results)
        return results