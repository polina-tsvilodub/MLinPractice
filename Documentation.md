## Feature Extraction

### Binary Feature Extraction

**Motivation**

There are some aspects of a tweet where it is not necessary to know how much of that aspect is in the tweet or what specific content the aspect has.
Instead, it is just interesting to know if the aspect is there or not. 
Therefore, we decided to implement a class that produces binary features from the dataset.

**Implementation**
The binary feature eextractor is implemented in the class `BinaryFeatureExtractor` which is located in `code/feature_extraction/binary_features.py`.
The class implements the method `_get_values()` which extracts the binary features from the given column of the dataset.

## Evaluation Schema
This project uses 5-fold cross validation to train and evaluate the classifiers.
Specifically, we use stratified cross validation to make sure that each fold is a good representative of the whole data.
Additinally, by using cross validation we use our whole data set to train and test our classifier.

## Baseline
We decided to use a stratified (a.k.a. label frequency based) baseline classifier to account for the embalance of the labels.
A stratified dummy classifier should be more robust in regard to this and therefore is a more challenging baseline to surpass.
Using this dummy classifier we also want to test whether "fancy" features are necessary or not for the classification. 