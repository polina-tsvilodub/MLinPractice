## Preprocessing steps

### Tokenization

**Motivation**

We decided to use a tokenizer during preprocessing since this is a essential step in making text machine interpretable.
The specific tokenizer we used is the `nltk TweetTokenizer`, it is a tokenizer specifically designed for tokenizing tweets which is the domain of this task.
We implemented the tokenizer with the following parameters:
- `preserve_case=False`: downcases all characters except for emoticons, saves us from implementing this separately
- `reduce_len=True`: limits repeating sequences to length of 3
- `strip_handles=True`: strips handles (@ mentions) from tweet as they are already documented in a separate column of the dataset

**Implementantion**

The tokenizer is implementned in `tokenizer.py` which contains the class `Tokenizer` which is a sublcass of `Preprocessor`.
The sublcass implements the method `_get_values()` which tokenizes a list of strings and outputs a list of lists of strings containing the tokenized tweets.


### Stopword Removal

**Motivation**

Stopwords are abundant and hence provide little to no unique information that can be used for classification.
Therefore, we decided to remove them during the preprocessing procedure to yield a better performance for our classifier.

**Implementation**
The stopword removal takes place after the tweet was already tokenized. We used the list of stopwords from nltk's `corpus` to decide which words are classified as stopwords.
The stopword remover is implemented in `code/preprocessing/stopword_remover.py` that contains the class `StopwordRemover` which is a subclass of `Preprocessor`.
The subclass implements the method `_get_values(self)` which removes the stopwords from the already tokenized tweets (list of lists of strings) and outputs the result the tweets containing no stopwords (list of lists of strings).

## Feature Extraction

### Binary Feature Extraction

**Motivation**

There are some aspects of a tweet where it is not necessary to know how much of that aspect is in the tweet or what specific content the aspect has.
Instead, it is just interesting to know if the aspect is there or not. 
Therefore, we decided to implement a class that produces binary features from the dataset, indicating the presence (or absense) of a given aspect.

**Implementation**
The binary feature eextractor is implemented in the class `BinaryFeatureExtractor` which is located in `code/feature_extraction/binary_features.py` and is a subclass of `FeatureExtractor`.
The class implements the method `_get_values()` which extracts the binary features from the given column of the dataset.

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