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


## Evaluation Schema
This project uses 5-fold cross validation to train and evaluate the classifiers.
Specifically, we use stratified cross validation to make sure that each fold is a good representative of the whole data.
Additinally, by using cross validation we use our whole data set to train and test our classifier.

## Baseline
We decided to use a stratified (a.k.a. label frequency based) baseline classifier to account for the embalance of the labels.
A stratified dummy classifier should be more robust in regard to this and therefore is a more challenging baseline to surpass.
Using this dummy classifier we also want to test whether "fancy" features are necessary or not for the classification. 