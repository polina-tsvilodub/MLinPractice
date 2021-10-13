## Feature Extraction

### Datetime Extraction

**Motivation**
Date and time of the publication of a tweet can influence its virality immensely. Therefore, we decided to use the publication datetime provided in the dataset as a feature.
However, the date and time as provided are not interpretable for a machine learning algorithm, they need to be converted into numerical values.
We decided to encode the month (1-12), the day of the month (1-31), and the hour (0-23) and the minutes (0-59) of a tweet's publication. The year was discarded due to not being a suitable predictive feature.
The year of the publication won't repeat and our classifier should be able to generalize to tweet's in the future.

**Implementation**
The date and time extractioni are implemented separately in the classes `DateExtractor` and `TimeExtractor` respectively.
The classes are both located in the same file `code/feature_extraction/datetime_extractor.py`. 
Both are a subclass of `FeatureExtractor` and implement the method `_get_values()` which extracts the desired date and time features as integers.

## Evaluation Schema
This project uses 5-fold cross validation to train and evaluate the classifiers.
Specifically, we use stratified cross validation to make sure that each fold is a good representative of the whole data.
Additinally, by using cross validation we use our whole data set to train and test our classifier.

## Baseline
We decided to use a stratified (a.k.a. label frequency based) baseline classifier to account for the embalance of the labels.
A stratified dummy classifier should be more robust in regard to this and therefore is a more challenging baseline to surpass.
Using this dummy classifier we also want to test whether "fancy" features are necessary or not for the classification. 