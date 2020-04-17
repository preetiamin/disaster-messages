import sys
import pandas as pd
from sqlalchemy import create_engine
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

import re
import nltk
nltk.download(['punkt', 'wordnet','stopwords'])
nltk.download('averaged_perceptron_tagger')
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

def tokenize(text):
    '''
    
    Processes and tokenizes text using following steps:
    - Removes hyperlink
    - Normalizes to lowercase
    - Removes words, alphanumeric words and punctuation
    - Tokenizes and tags words with parts of speech
    - Uses parts of speech to lemmatize words
    
    Parameter: text to be tokenized
    Returns: list of tokens
    
    '''
    # remove url and add placeholder instead
    text = re.sub('http[s]?\s*:\s*\S+','urlplaceholder',text)
    text = re.sub('http[s]?\s*\S+\s*\S*','urlplaceholder',text)

    # normalize case, remove numbers, alphanumeric words and punctuation
    text = text.lower()
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    
    # Tokenize and lemmatize based on pos_tag, ignore words 2 letters or less and stop words
    lemmatizer = WordNetLemmatizer()
    tokens = []
    for word,tag in pos_tag(word_tokenize(text)):
        if (len(word)>2) and (word not in stopwords.words("english")):
            tag = tag[0].lower()
            tag = tag if tag in ['a','r','n','v'] else None
            if not tag:
                tokens.append(word)
            else:
                tokens.append(lemmatizer.lemmatize(word,tag))
    return tokens

def load_data(database_filepath):
    '''
    
    Loads data from a sqlite database and saves it to a pandas dataframe. Assigns X and Y to be used 
    to model the data
    
    Parameters: 
    database_filepath: Name of the sqlite database
    Returns:
    X: dataframe containing single column of input text data
    Y: dataframe containing the 36 columns to be used as the output
    
    '''
    
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql("SELECT * FROM Messages", engine)
    df['message']=df['message'].apply(lambda x: x +' question' if '?' in x else x)
    X = df['message']
    Y = df.iloc[:,4:]
    return X, Y, list(Y.columns)

def build_model():
    '''
    
    Creates a pipeline for the nlp model using CountVectorizer, TF-IDF Transformer and Mulit Output
    Classifier using Random Forest Classifier
    
    Parameters: None
    Returns: Pipeline
    
    '''
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize, max_df=0.5, max_features=5000, ngram_range=(1,2))),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    
    Performs a prediction on test data and outputs classification report for each output
    
    Parameters:
    model: Model to be used to fit the data
    X_test: Input data to be used for prediction
    Y_test: Actual labels for the data
    
    Returns: None
    
    '''
    Y_pred = model.predict(X_test)
    Y_pred_df = pd.DataFrame(Y_pred,columns=category_names)
    for col in category_names:
        print(col,classification_report(Y_test[col],Y_pred_df[col]))


def save_model(model, model_filepath):
    
    '''
    
    Saves model to a pickle file
    
    Parameters:
    model: Model to be saved
    model_filepath: name to pickle file to be saved to
    
    Returns: None
    '''
    # load model
    joblib.dump(model, model_filepath)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()