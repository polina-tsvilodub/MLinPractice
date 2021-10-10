## Preprocessing steps

### Stopword Removal

**Motivation**

Stopwords are abundant and hence provide little to no unique information that can be used for classification.
Therefore, we decided to remove them during the preprocessing procedure to yield a better performance for our classifier.

**Implementation**
The stopword removal takes place after the tweet was already tokenized. We used the list of stopwords from nltk's `corpus` to decide which words classified as stopwords.
Thre stopword remover is implemented in `code/preprocessing/stopwrod_remover.py` which contains the class `StopwordRemover` which is a subclass of `Preprocessor`.
The subclass implements the method `_get_values(self)` which removes the stopwords from the already tokenized tweets (list of lists of strings) and outputs the result the tweets containing no stopwords (list of lists of strings).


## Evaluation Schema
This project uses 5-fold cross validation to train and evaluate the classifiers.
Specifically, we use stratified cross validation to make sure that each fold is a good representative of the whole data.
Additinally, by using cross validation we use our whole data set to train and test our classifier.

## Baseline
We decided to use a stratified (a.k.a. label frequency based) baseline classifier to account for the embalance of the labels.
A stratified dummy classifier should be more robust in regard to this and therefore is a more challenging baseline to surpass.
Using this dummy classifier we also want to test whether "fancy" features are necessary or not for the classification. 