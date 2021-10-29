#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 05:33:00 2021

@author: lschiesser and ptsvilodub
"""

# Documentation of the Tweets-Going-Viral Project by group *Ultimate Power Touple 00101010*

This project is conducted as part of the block seminar "Machine learning in practice" at the Osnabrück University in WS2021/22. 
The goal of this project is to implement a classifier predicting whether a tweet containing either of the keywords "data analysis", "data science" or "data visualization" will go viral. To this end, the notion of going viral is operationalized as a tweet having a total of more than 50 likeas and retweets in sum. 
Furthermore, the goal of this project is to develop the classifier in a team while adhering to industrial best practices of agile software engineering and data science. 

Below, the implementation of the project is documented, mirroring the classification pipeline.  

The four major pipeline steps are:
* Data preprocessing
* Feature extraction
* Classifier training
* Classifier evaluation 

For each step, we document the specific modules executing the step; for each module, its motivation and implementation are described.
Finally, we describe and discuss our results.

## General set-up instructions

The overall execution flow is briefly described as an introduction, before going into details of the single modules. 

Each major module is implemented in a separate directory. In each directory, there is a `.sge` script created for running the pipeline on the IKW computing grid. An analogous `.sh` script for running the pipeline locally can be found in the code directory. 
The options of the pipeline steps are implemented as command line arguments and are, therefore, passed in the `.sge` scripts. This allows to flexibly adapt the pipeline components. 

Due to runtime constraints provided by the IKW grid, we created two separate preprocessing job scripts (see below for details). Due to the same reason we created three separate feature extraction job scripts (one per training, validation and test set, respectively).  

Ideally, one could start the entire pipeline from the `code/pipeline.sge` (or .sh) script, but this is not recommended for the grid due to long runtime. The user is advised to submit one job per module `.sge` script. 

Finally, in order to make predictions with the final trained classifier one can use the script `code/application.sge` (.sh) which allows to interactively enter tweets and receive virality predictions for them.


## Preprocessing steps

First, the input data is preprocsessed. This is a standard first step of machine learning pipelines aiming to massage data into the required shape and maximally improve its quality by removing noise.
The specific measures to do so depend on the nature of the data. Since our input data consists of tweets (i.e., texts) and their meta information, we focus on common text preprocessing techniques.
All preprocessing steps are implemented in `code/preprocessing`. 

### Label creation

Before preprocessing, labels are added to the data according to the assumed operationalization of virality. That is, a new binary column "label" is created which contains the boolean TRUE if the sum of the entries from the columns "likes_count" and "retweets_count" is above 50, and FALSE otherwise. 
The script `code/preprocessing/create_labels.py` is called from `code/preprocessing/preprocessing_tokenize.sge`.

### Punctuation, emoji and special charcaters removal

In this step, specific characters or character sequences are removed. These characters could potentially corrupt embeddings (see below) while carrying little information, and, therefore, are commonly removed. 
This is the first preprocessing step, executed before the tweets are tokenized. 

**Motivation**

As a further preprocessing step, we implement a class for removing punctuation, emojis and special characters from the raw input tweets. We decided on this preprocessing step because it removes elements which are difficult for encoding as embeddings. 
That is, words containing special characters or emojis usually are not recognized as words and decrease the quality of the embeddings of the texts. Since we want to encode the tweets as embeddings, we include this preprocessing step to increase embedding quality un subsequent feature extraction steps.  

**Implementation**

In `code/preprocessing` we implement `punctuation_remover.py` which contains the class `PunctuationRemover` which is a subclass of `Preprocessor`.
The subclass implements the following method(s):
* `_set_variables(self, inputs):` Set punctuation in `self._punctuation` that will be removed by the `_get_values`method.  
* `deEmojify(text):` Removes unicode encoded emojis from a string. 
* `_get_values(self, inputs):` Remove punctuation, emojis and special characters from the input column containing a list of sentence strings. 
  
  Returns a column containing a list of clean sentence strings.  

### Tokenization

In this step, text strings are split into smaller units. We decided to implement word-level tokenization, i.e., tweet strings are split into single words. This was done to allow removing single tokens like stop words (see below).
This and the previous steps are initialized in `code/preprocessing/preprocessing_tokenize.sge`.

**Motivation**

We decided to use a tokenizer during preprocessing since this is an essential step in making text machine interpretable.
The specific tokenizer we used is the `nltk TweetTokenizer`; it is a tokenizer specifically designed for tokenizing tweets which is the domain of this task.

**Implementantion**

The tokenizer is implementned in `code/preprocessing/tokenizer.py` which contains the class `Tokenizer`. It is a sublcass of the `Preprocessor` class.
The sublcass implements the method `_get_values()` which tokenizes a list of strings.
We implemented the tokenizer with the following parameters:
- `preserve_case=False`: downcases all characters except for emoticons, saves us from implementing lowercasing separately.
- `reduce_len=True`: limits repeating sequences to length of 3.
- `strip_handles=True`: strips handles (@ mentions) from tweets as they are already documented in a separate column of the dataset.

  Returns a list of lists of strings containing the tokenized tweets

### Stopword Removal

In this step, specific tokens are removed from tokenized sentences. We create a custom lists of nine stopwords which we deem most frequent, based on nltk's `corpus` of stopwords. We do not use off-the-shelf stopwords lists because iterating over them yields too long runtimes. This step is initialized in `code/preprocessing/preprocessing_stopwords.sge`.

**Motivation**

Stopwords are abundant and hence provide little to no unique information about the content of a tweet that could be used for classification.
Therefore, we decided to remove them during the preprocessing procedure to yield a better performance for our classifier.

**Implementation**

The stopword removal takes place after the tweet was already tokenized. 
The stopword remover is implemented in `code/preprocessing/stopword_remover.py` that contains the class `StopwordRemover` which is a subclass of `Preprocessor`.
The subclass implements the method `_get_values()` which removes the stopwords from the already tokenized tweets (list of lists of strings).

Returns the resulting tweets containing no stopwords (list of lists of strings).

### Data splits

Finally, the preprocessed dataset is split into training, validation and test datasets according to a 60 : 20 : 20 split (which can be customized). This is implemented in `code/preprocessing/split_data.py`.   
Features required for the classifier are then extracted separately for each dataset split.   
  
## Feature Extraction

Below, we document the features we extract and feed into our classifier. The selection of these features is based on the list brainstormed during the respective seminar session, as well as on our general knowledge of useful features in classification tasks.
All feature extraction steps are implemented in `code/feature_extraction`.

### Binary Feature Extraction

A binary feature can only take one of two values, represented by a 1D one-hot encoded vector. 

**Motivation**

There are some aspects of a tweet where it is not necessary to know how much of that aspect is in the tweet or what specific content the aspect has.
Instead, it is just interesting to know if the aspect is there or not. 
Therefore, we decided to implement a class that produces binary features from the dataset, indicating the presence (or absense) of a given aspect.
This feature was specifically design for indicating the presence of media or links in the tweet.

**Implementation**

The binary feature eextractor is implemented in the class `BinaryFeatureExtractor` which is located in `code/feature_extraction/binary_features.py` and is a subclass of `FeatureExtractor`.
The class implements the method `_get_values()` which extracts the binary features from the given column of the dataset.

Returns a column of one-hot encoded binary numpy arrays.

### Numerical Feature Extraction

A numerical feature represents a continuous value of some variable. 

**Motivation**

Some aspects of a tweet are encoded separately from the tweet in the dataset, e.g., the hashtags and @ mentions. 
Since it is not necessarily interesting who or which hashtag was mentioned but how many are present, it is of interest to compute the number of hashtags or the number of @ mentions. 
Therefore, we implement a feature extractor that computes these features. 
The features are all represented as lists in the dataset. 
That is, the length of the list is the number of hashtags or @ mentions.

**Implementation**

The numerical feature extractor is implemented in the class `NumericalFeatureExtractor` which is located in `code/feature_extraction/numerical_featuers.py`.
The class implements the method `_get_values()` which extracts the length of each nested list returning a list of integer. 

Returns a list of integers.

### Datetime Extraction

This feature parses the date and time information into a machine readable format.

**Motivation**

Date and time of the publication of a tweet can influence its virality immensely. Therefore, we decided to use the publication datetime provided in the dataset as a feature.
However, the date and time as provided are not interpretable for a machine learning algorithm, so they need to be converted into numerical values.
We decided to encode the month (1-12), the day of the month (1-31), the hour (0-23) and the minutes (0-59) of a tweet's publication. The year was discarded due to not being a suitable predictive feature.
The year of the publication won't repeat and our classifier should be able to generalize to tweets in the future.

**Implementation**

The date and time extraction are implemented separately in the classes `DateExtractor` and `TimeExtractor`, respectively.
The classes are both located in the same file `code/feature_extraction/datetime_extractor.py`. 
Both are a subclass of `FeatureExtractor` and implement the method `_get_values()` which extracts the desired date and time features as integers.

Return lists of integers, respectively.

### Text embeddings

This feature converts strings to numerical arrays in order to represent them in a machine readable format. 

**Motivation**

The main feature of our data is the actual content, i.e., text of the tweet to be categorized. State-of-the-art representations of text and its meaning are so-called embeddings (e.g., Mikolov et al., 2013). Therefore, we decided to use embeddings in order to represent the tweet.
Since creating (i.e., training) embeddings from scratch is quite costly, we decided to use available pretrained solutions. More specifically, we decided to use pretrained 25-dimensional GloVe embeddings provided by the `gensim` package, due to their easily accessible API.
But, more importantly, the `gensim` GloVe emebddings were created using a Twitter dataset, which makes them a very suitable choice for our use case. Furthermore, we decided to use relatively low-dimensional embeddings in order to avoid dimensionality issues during classification. 
The decision how to compute tweet-level embeddings from token-level embeddings is driven by manageability and tractability reasons; we are aware that in more advanced embedding techniques out-of-vocabulary words usually receive a dedicated token (see below).

**Implementation**

In `code/feature_extraction`, we implement `embeddings.py` which implements the `Embeddings` class, a subclass of `FeatureExtractor`. The subclass implements the following methods:
  *`_set_variables(self, inputs)`: Sets internal variable given input columns. Downloads GloVe embeddings, if not downloaded yet, pretrained on the `'glove-twitter-25'` dataset.
  * `_get_values(self, inputs)`: Computes tweet-level embeddings. This is done by extracting word-level embeddings (if `KeyError` raised, the words are just skipped). Then, tweet-level embeddings are computed as average of word-level embeddings for a given input column.
        Returns a numpy array of tweet-level embeddings.
      

## Classification

The classifier is the core of our application, since it completes the actal task of predicting whether a tweet goes viral, given its feature values. There are quite many approaches and architectures for classification, and the choice of a particular type is usually motivated by the nature of the data and the task.
All classifier related modules are implemented in `code/classification`.
                                                                                                                          
**Motivation**

Given that our task involves binary classification, several classifier types like logistic regression, neural networks or support vector machines (SVM) were in question. We decided to use an SVM as our final classifier architecture due to the following reasons:
    * training a neral network might require even more data and is computationally quite expensive. Furthermore, optimizing the architecture of the network itself goes beyond the scope of such a project and might be an overkill for the task.
    * logistic regression is limited in that it only makes use of a linear combination of the features in logit space. This linearity assumption might be too strong when dealing with textual data.
    * random forest classifiers are not well-suited for data containing continuous features.
    * given general experience, an SVM is usually a good choice for a basic but robust classifier. It is computationally easier and might use a non-linear projection kernel, such that it might be better suitable for higher-dimensional data. From experience, it also performs reasonably well on textual data.

Last but not least, sklearn also provides an simple API for using SVMs. For training the SVM, we chose to use sklearn's `GridSearchCV` method (see "Evaluation schema" below). 
The classifier with the best-performing hyperparameters which is refit on the entire training dataset is then used for the downstream steps.

**Implementation**

The classifier is implemented in `code/classification/run_classifier.py`. The file implements two classifiers:
    * the main SVM classifier as an instance of `sklearn.svm.LinearSVC`
        * given that the size of our training dataset exceeds tens of thousands of samples, the more efficient `LinearSVC` implementation was used (not the basic `SVC` classifier).
        * the classifier is implemented with the parameter `dual=False` because our number of features is smaller than the number of samples
        * the SVM is ft using `GridSearchCV` (see "Evaluation schema" below)
    * the baseline dummy classifier as an instance of `sklearn.dummy.DummyClassifier` (see below)
    * the computation of an evaluation metrics suite for each of the classifiers (see below)
        * evaluation results are stored in `.csv` files in the `results/` directory. 

      Returns a trained sklearn classifier instance.
      
## Evaluation Metrics

The classifier needs to be evaluated in order to assess how well it generalizes to predicting virality of unseen tweets. There are many different evaluation techniques, different with respect to their interpretability in different use cases. 
 
**Motivation**

We decided to implement the following evaluation metrics for our project:
* standard accuracy (as provided by the departure point code): proportion of correctly identified labels. This metric is used in many projects and can, therefore, be interesting for comparison purposes.
* balanced accuracy (accuracy score compensated for imbalanced datasets): average of recall per class. This metric is better suited for class imbalanced datasets, whereby our dataset is an instance thereof.
* F1-score (a score combining precision and recall): it is more representative than just the accuracy due to taking into account both correct identifications as well as successes of getting the underrepresented class.
* Cohen's kappa: a more robust score for imblanaced datasets; it is also called interrater reliability score and accounts for the probability of correct classification at random.
* ROC (Receiver Operating Characteristic (curve)): outputs AUC (area under the curve) computed on the curve resulting from plotting the true positive rate of the classifier against its false positive rate. 

As outlined above, each metric has specific advantages and disadvantages, respectively. This suite of evaluation metrics allows the user to get a comprehensive picture of the classifier performance, given diverse information. 
In general, the user is adviced to use several metrics and interpret them in combination in order to gain a deeper understanding of the quality of the classifier. 

**Implementation**

The suite of evaluation metrics is specified in `code/util.py/EVAL_METRICS` as a dictionary of the shape `{"metric_name": sklearn_metric_function_name}`.
The user is welcome to update this dictionary.

In `code/evaluation/` , we implement `evaluation_metrics.py` which contains the class `EvaluationMetrics`.
The class has the following method:
* `compute_metrics(self, y_true, y_pred, metrics=EVAL_METRIC)`: computes all evaluation metrics specified in `metrics` and creates a pd.DataFrame of the results, given vectors containing the ground truth and the classifier predictions. The results are stored into the default directory `results/`.

  Returns a pd.Dataframe.
  
## Evaluation Schema 

**Motivation** 

This project uses 5-fold cross validation to train and evaluate the classifier.
The number of folds can also be changed (see README.md). Although the dataset at hand might be large enough to just use one iteration of splitting and training, we decided to use cross validation in order to decrease the risk of introducing potential artifacts due to the imbalanced label distribution.
For the same reason, we use stratified cross validation to make sure that each fold is a good representative of the whole data (i.e., the distriution of classes is taken into account during splitting).  
Additinally, by using cross validation, we make maximal use of our data set to train and test our classifier.

Furthermore, the SVM classifier has some hyperparameters (i.e., parameters of the classifier which don't depend on the data) which need to be tuned. A common algorithm for choosing those is grid search. 
We decided to use its vanilla version because we perform the grid search on quite a small range of parameters, given the runtime constraints of the IKW grid. It is rather a proof of concept, such that the parameter space and the respective computational constraints aren't important. 
Therefore, we use the method GridSearchCV which conveniently combines both training aspects in a wrapper.

**Implementation**

The cross validation implementation is integrated with grid search over hyper parameters of the SVM production classifier, as implemented by the sklearn method `GridSeachCV`.
In `code/classification/run_classifier.py`, the evaluation schema is implemented as part of the training of the classifier:
* the classifier is trained using the `sklearn.model_selection.GridSearchCV.fit()` method. 
    * the parameters over which the grid search is performed are implemented in `code/classification/run_classifier.py` as a distionary `parameters = {"penalty": ("l1", "l2"), "C": [1,2], "dual": (False)}`. This selection is motivated by the parameter combinations supported by `LinearSVC` and serve as a proof of concept for runtime reasons.                                                        
* the final classifier is accessed and dumped via the method's attribute `best_estimator_`.
    * the evaluation metrics described above are computed on this best performing fitted production classifier.
    
## Evaluation Baseline

**Motivation**

We decided to use a stratified (a.k.a. label frequency based) baseline classifier to account for the embalance of the labels.
A stratified dummy classifier should be more robust in regard to this and, therefore, is a more challenging baseline to surpass.
Using this dummy classifier we also want to test whether "fancy" features are necessary or not for the classification. 

**Implementation**

The dummy classifier is also implemented in `code/classification/run_classifier.py`:
    * the baseline dummy classifier is an instance of `sklearn.dummy.DummyClassifier`
        * the parameter `strategy = "stratified"` is used                                                  
    * the evaluation metrics suite is computed for the baseline  
        * evaluation results are stored in `.csv` files in the `results/` directory 


## Results & Discussion 

???
