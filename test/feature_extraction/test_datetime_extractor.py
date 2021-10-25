#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 11:28:14 2021

@author: lschiesser
"""

import unittest
import pandas as pd
from code.feature_extraction.datetime_extractor import DateExtractor, TimeExtractor

class TestDatetimeExtractor(unittest.TestCase):
    
    def setUp(self):
        self.INPUT_COLUMN_DATE = "input"
        self.OUTPUT_COLUMNS_DATE = ["month", "day"]
        self.INPUT_COLUMN_TIME = "input"
        self.OUTPUT_COLUMNS_TIME = ["hour", "minute"]
        
        self.date_extractor = DateExtractor(self.INPUT_COLUMN_DATE, self.OUTPUT_COLUMNS_DATE)
        self.time_extractor = TimeExtractor(self.INPUT_COLUMN_TIME, self.OUTPUT_COLUMNS_TIME)
    
    
    def test_date_output_column(self):
        self.assertEqual(self.date_extractor.get_feature_name(), self.OUTPUT_COLUMNS_DATE)
    
    def test_time_output_column(self):
        self.assertEqual(self.time_extractor.get_feature_name(), self.OUTPUT_COLUMNS_TIME)

    
    def test_date_extractor(self):
        date = "2021-04-14"
        # the expected output is a np array, but unittest can't compare np arrays
        # therefore, transform np array from class output to list and comopare with this list
        expected_output = [4, 14]
        # create input dataframe df
        df = pd.DataFrame()
        df[self.INPUT_COLUMN_DATE] = [date]
        output = self.date_extractor.fit_transform(df)
        # use assertListEqual from unittest to compare output (numpy array cast to list) and list
        self.assertListEqual(output[0].tolist(), expected_output)
    
    def test_time_extractor(self):
        time="4:46:42"
        # the expected output is a np array, but unittest can't compare np arrays
        # therefore, transform np array from class output to list and comopare with this list
        expected_output = [4, 46]
        
        # create input dataframe df
        df = pd.DataFrame()
        df[self.INPUT_COLUMN_TIME] = [time]
        output = self.time_extractor.fit_transform(df)
        # use assertListEqual from unittest to compare output (numpy array cast to list) and list
        self.assertListEqual(output[0].tolist(), expected_output)


if __name__ == '__main__':
    unittest.main()