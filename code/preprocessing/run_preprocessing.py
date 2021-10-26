#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runs the specified collection of preprocessing steps

Created on Tue Sep 28 16:43:18 2021

@author: lbechberger
"""

import argparse, csv, pickle
import pandas as pd
from sklearn.pipeline import make_pipeline
from code.preprocessing.punctuation_remover import PunctuationRemover
from code.preprocessing.stopword_remover import StopwordRemover
from code.preprocessing.tokenizer import Tokenizer
from code.preprocessing.stemmer import Stemmer
from code.util import SUFFIX_STEMMED, TWEET_TOKENIZED

# setting up CLI
parser = argparse.ArgumentParser(description = "Various preprocessing steps")
parser.add_argument("input_file", help = "path to the input csv file")
parser.add_argument("output_file", help = "path to the output csv file")
parser.add_argument("-p", "--punctuation", action = "store_true", help = "remove punctuation")
parser.add_argument("-e", "--export_file", help = "create a pipeline and export to the given location", default = None)
parser.add_argument("-sw", "--stopwords", help="remove stopwords from tokenized tweets", action="store_true")
parser.add_argument("-st", "--stemming", help = "stem tokenized sentences", action = "store_true")
parser.add_argument("--stemming_input", help = "input column of tokenized sentence lists for stemming", default = TWEET_TOKENIZED)
parser.add_argument("-t", "--tokenize", help = "tokenize each sentence", action="store_true")

args = parser.parse_args()

# load data
df = pd.read_csv(args.input_file, quoting = csv.QUOTE_NONNUMERIC, lineterminator = "\n")

# collect all preprocessors
preprocessors = []
if args.punctuation:
    preprocessors.append(PunctuationRemover())

if args.tokenize:
    preprocessors.append(Tokenizer())

if args.stopwords:
    preprocessors.append(StopwordRemover())

if args.stemming:
    preprocessors.append(Stemmer(args.stemming_input, args.stemming_input + SUFFIX_STEMMED))
    
# call all preprocessing steps
for preprocessor in preprocessors:
    df = preprocessor.fit_transform(df)

# store the results
df.to_csv(args.output_file, index = False, quoting = csv.QUOTE_NONNUMERIC, line_terminator = "\n")

# create a pipeline if necessary and store it as pickle file
if args.export_file is not None:
    pipeline = make_pipeline(*preprocessors)
    with open(args.export_file, 'wb') as f_out:
        pickle.dump(pipeline, f_out)