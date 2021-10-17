## Feature Extraction
### Numerical Feature Extraction

**Motivation**

Some aspects of a tweet are encoded separately from the tweet in the dataset, e.g. the hashtags and @ mentions. 
Since it is not necessarily interesting who or which hashtag was mentioned but how many, it is of interest to compute the number of hashtags or the number of @ mentions. 
Therefore, we implement a feature extractor that computes these features. 
The features are all represented as lists in the dataset. 
Therefore, the length of the list is the number of hashtags or @ mentions.

**Implementation**

The numerical feature extractor is implemented in the calss `NumericalFeatureExtractor` which is located in `code/feature_extraction/numerical_featuers.py`.
The class implements the method `_get_values()` which extracts the length of the nested list returning an integer. 
## Evaluation Schema
This project uses 5-fold cross validation to train and evaluate the classifiers.
Specifically, we use stratified cross validation to make sure that each fold is a good representative of the whole data.
Additinally, by using cross validation we use our whole data set to train and test our classifier.

## Baseline
We decided to use a stratified (a.k.a. label frequency based) baseline classifier to account for the embalance of the labels.
A stratified dummy classifier should be more robust in regard to this and therefore is a more challenging baseline to surpass.
Using this dummy classifier we also want to test whether "fancy" features are necessary or not for the classification. 