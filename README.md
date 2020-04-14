# disaster-messages
Categorize disaster messages from tweets

https://medium.com/@preeti.amin/?

## Introduction
A machine learning model was built to classify disaster tweets from a provided dataset. The tweets were categorized into 36 different categories, where each tweet could be categorized into multiple categories.

## Project Motivation
The project was completed as part of Udacity's Data Scientist nanodegree program. The purpose of the project was to gain understanding in natural language processing.

## Methods Used
* Customer Tokenizer - A custom tokenizer function was written to tokenize the text. This function removes all hyperlinks, numbers and alphanumeric strings as well as all punctuation. It also normalizes the text to lowercase. It further tokenizes the text and tags it with parts of speech which are used to lemmatize the text.

* Model Pipeline - A pipeline was built from Count Vectorizer using the custom tokenizer followed by the TF-IDF transformer, followed by a Multi Output Classifier using Random Forest Classifier as the estimator. With the exception of the custom tokenizer, all other values were left as default.

* Grid Search - Grid Search was used to optimize the model. Four paramters were optimized using the grid search:
- Count Vectorizer ngram_range: Values of (1,1) and (1,2) were used, ngram_range of (1, 1) means only unigrams, (1, 2) means unigrams and bigrams were used in the data; default is (1,1) which means only unigrams are used.
- Count Vectorizer max_df: Values of 0.5 and 1.0 were used, words with frequency higher than max_df are ignored when building the vocabulary; default is 1.0, which means nothing is ignored.
- Count Vectorizer max_features: Values of None and 10,000 were used, when building a vocabulary, the max features are bounded by this argument; default is None, which mean all features are used.
- TF-IDF Transformer use_idf: Values of True and False were used, when building the inverse document frequency, setting this value of True down-weights the higher frequency words; default is True.

The optimal parameters found using the grid search were max_df of 0.5, max_features 10,000 and n_gram_range of (1,2) for Count Vectorizer and use_idf=True for TF-IDF Transformer.

## Results

## Conclusion

## Acknowledgements

