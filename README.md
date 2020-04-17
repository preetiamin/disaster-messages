## Introduction
A machine learning model was built to classify disaster tweets from a provided dataset. The tweets were categorized into 36 different categories, where each tweet could be categorized into multiple categories.

## Project Motivation
The project was completed as part of Udacity's Data Scientist nanodegree program. The purpose of the project was to gain understanding in natural language processing using a pipeline and grid search.

## Methods Used
#### Customer Tokenizer
A custom tokenizer function was written to tokenize the text. This function removes all hyperlinks, numbers and alphanumeric strings as well as all punctuation. It also normalizes the text to lowercase. It then tokenizes the text and tags it with parts of speech which are used to lemmatize the text.

#### Model Pipeline
A pipeline was built from Count Vectorizer using the custom tokenizer followed by the TF-IDF transformer, followed by a Multi Output Classifier using Random Forest Classifier as the estimator. With the exception of the custom tokenizer, all other values were left as default.

#### Grid Search
Grid Search was used to optimize the model. Four paramters were optimized using the grid search:
- Count Vectorizer ngram_range: Values of (1,1) and (1,2) were used, ngram_range of (1, 1) means only unigrams, (1, 2) means unigrams and bigrams were used in the data; default is (1,1) which means only unigrams are used.
- Count Vectorizer max_df: Values of 0.5 and 1.0 were used, words that appear in more than max_df proportion of documents are ignored when building the vocabulary; default is 1.0, which means nothing is ignored.
- Count Vectorizer max_features: Values of None and 10,000 were used, when building a vocabulary, the max features are bounded by this argument; default is None, which mean all features are used.
- TF-IDF Transformer use_idf: Values of True and False were used, when building the inverse document frequency, setting this value of True down-weights the higher frequency words; default is True.

## Results

The optimal parameters found using the grid search were max_df of 0.5, max_features 10,000 and n_gram_range of (1,2) for Count Vectorizer and use_idf=True for TF-IDF Transformer. Another run using max_features of 5,000 was performed which showed similar performance, hence 5,000 was used for the model.

Below is the classifiction report for the model:

related              precision    recall  f1-score   support

          0       0.58      0.49      0.53      1215
          1       0.85      0.89      0.87      4029

avg /total       0.79      0.80      0.79      5244

request              precision    recall  f1-score   support

          0       0.90      0.97      0.94      4358
          1       0.79      0.47      0.59       886

avg / total       0.88      0.89      0.88      5244

/opt/conda/lib/python3.6/site-packages/sklearn/metrics/classification.py:1135: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
offer              precision    recall  f1-score   support

          0       1.00      1.00      1.00      5219
          1       0.00      0.00      0.00        25

avg / total       0.99      1.00      0.99      5244

aid_related              precision    recall  f1-score   support

          0       0.77      0.84      0.81      3108
          1       0.74      0.64      0.69      2136

avg / total       0.76      0.76      0.76      5244

medical_help              precision    recall  f1-score   support

          0       0.94      0.99      0.96      4862
          1       0.58      0.16      0.25       382

avg / total       0.91      0.93      0.91      5244

medical_products              precision    recall  f1-score   support

          0       0.96      1.00      0.98      4993
          1       0.79      0.18      0.29       251

avg / total       0.95      0.96      0.95      5244

search_and_rescue              precision    recall  f1-score   support

          0       0.98      1.00      0.99      5111
          1       0.53      0.08      0.13       133

avg / total       0.97      0.97      0.97      5244

security              precision    recall  f1-score   support

          0       0.98      1.00      0.99      5163
          1       0.17      0.01      0.02        81

avg / total       0.97      0.98      0.98      5244

military              precision    recall  f1-score   support

          0       0.97      1.00      0.99      5091
          1       0.57      0.08      0.15       153

avg / total       0.96      0.97      0.96      5244

child_alone              precision    recall  f1-score   support

          0       1.00      1.00      1.00      5244

avg / total       1.00      1.00      1.00      5244

water              precision    recall  f1-score   support

          0       0.97      0.99      0.98      4909
          1       0.81      0.56      0.66       335

avg / total       0.96      0.96      0.96      5244

food              precision    recall  f1-score   support

          0       0.96      0.98      0.97      4654
          1       0.82      0.68      0.74       590

avg / total       0.94      0.95      0.94      5244

shelter              precision    recall  f1-score   support

          0       0.95      0.99      0.97      4778
          1       0.77      0.43      0.55       466

avg / total       0.93      0.94      0.93      5244

clothing              precision    recall  f1-score   support

          0       0.99      1.00      1.00      5171
          1       0.87      0.37      0.52        73

avg / total       0.99      0.99      0.99      5244

money              precision    recall  f1-score   support

          0       0.98      1.00      0.99      5122
          1       0.75      0.05      0.09       122

avg / total       0.97      0.98      0.97      5244

missing_people              precision    recall  f1-score   support

          0       0.99      1.00      0.99      5185
          1       0.50      0.03      0.06        59

avg / total       0.98      0.99      0.98      5244

refugees              precision    recall  f1-score   support

          0       0.97      1.00      0.98      5076
          1       0.39      0.09      0.15       168

avg / total       0.95      0.97      0.96      5244

death              precision    recall  f1-score   support

          0       0.97      1.00      0.98      5010
          1       0.78      0.27      0.40       234

avg / total       0.96      0.96      0.96      5244

other_aid              precision    recall  f1-score   support

          0       0.88      0.99      0.93      4555
          1       0.48      0.08      0.14       689

avg / total       0.83      0.87      0.83      5244

infrastructure_related              precision    recall  f1-score   support

          0       0.94      1.00      0.97      4917
          1       0.36      0.01      0.02       327

avg / total       0.90      0.94      0.91      5244

transport              precision    recall  f1-score   support

          0       0.96      1.00      0.98      4993
          1       0.62      0.10      0.18       251

avg / total       0.94      0.95      0.94      5244

buildings              precision    recall  f1-score   support

          0       0.96      1.00      0.98      4981
          1       0.75      0.21      0.33       263

avg / total       0.95      0.96      0.95      5244

electricity              precision    recall  f1-score   support

          0       0.98      1.00      0.99      5131
          1       0.80      0.07      0.13       113

avg / total       0.98      0.98      0.97      5244

tools              precision    recall  f1-score   support

          0       0.99      1.00      1.00      5210
          1       0.00      0.00      0.00        34

avg / total       0.99      0.99      0.99      5244

hospitals              precision    recall  f1-score   support

          0       0.99      1.00      1.00      5196
          1       0.00      0.00      0.00        48

avg / total       0.98      0.99      0.99      5244

shops              precision    recall  f1-score   support

          0       1.00      1.00      1.00      5225
          1       0.00      0.00      0.00        19

avg / total       0.99      1.00      0.99      5244

aid_centers              precision    recall  f1-score   support

          0       0.99      1.00      0.99      5186
          1       0.00      0.00      0.00        58

avg / total       0.98      0.99      0.98      5244

other_infrastructure              precision    recall  f1-score   support

          0       0.96      1.00      0.98      5011
          1       0.00      0.00      0.00       233

avg / total       0.91      0.95      0.93      5244

weather_related              precision    recall  f1-score   support

          0       0.89      0.95      0.91      3794
          1       0.83      0.68      0.75      1450

avg / total       0.87      0.87      0.87      5244

floods              precision    recall  f1-score   support

          0       0.96      0.99      0.98      4804
          1       0.90      0.51      0.65       440

avg / total       0.95      0.95      0.95      5244

storm              precision    recall  f1-score   support

          0       0.95      0.98      0.97      4762
          1       0.75      0.53      0.62       482

avg / total       0.94      0.94      0.94      5244

fire              precision    recall  f1-score   support

          0       0.99      1.00      0.99      5186
          1       0.86      0.10      0.18        58

avg / total       0.99      0.99      0.99      5244

earthquake              precision    recall  f1-score   support

          0       0.98      0.99      0.98      4755
          1       0.87      0.79      0.82       489

avg / total       0.97      0.97      0.97      5244

cold              precision    recall  f1-score   support

          0       0.99      1.00      0.99      5141
          1       0.79      0.25      0.38       103

avg / total       0.98      0.98      0.98      5244

other_weather              precision    recall  f1-score   support

          0       0.95      1.00      0.97      4973
          1       0.38      0.03      0.05       271

avg / total       0.92      0.95      0.93      5244

direct_report              precision    recall  f1-score   support

          0       0.86      0.97      0.91      4215
          1       0.71      0.35      0.47      1029

avg / total       0.83      0.84      0.82      5244

## Conclusion
Overall, RandomForestClassifier model worked well for this dataset with some limitations. AdaBoostClassifier was also tested as another model but performed only slightly better than RandomForestClassifier. 

Some categories did not have enough data to train the model well; further work into balancing the dataset might be helpful in boosting the accuracy of the model. Further work on assigning importance to categories might also be helpful. For example, detecting an earthquake accurately may be of much more importance than detecting category of tools. 

Categories could also be refined further as it's difficult to comprehend from the dataset what related, other_weather etc. are supposed to mean. But overall, the model perfomed fairly well in detecting various categories for the messages.

## Acknowledgements

https://www.udacity.com/
