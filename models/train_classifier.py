import sys
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
import re  # punctuation removal
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import sklearn.externals
import joblib


def load_data(database_filepath):
    ''' load data from sqlite database, output feature set, target and target categories

    Args:
        database_filepath: the path for the sqlite database.

    Returns:
        X: feature set.
        Y: target.
        category_names: target categories. 

    '''

    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Disaster_Response', engine)
    X = df['message']
    Y = df.drop(columns = ['id', 'message', 'original', 'genre'])
    category_names = Y.columns.values.tolist()

    return X, Y, category_names



def tokenize(text):
    ''' Write a tokenization function to process the text data

    Args:
        text: input text

    Returns:
        lemmed: the processed text data

    '''

    # case normalization: convert to lowercase
    text = text.lower()
    
    # punctuation removal
    text = re.sub(r"[^a-zA-Z0-9]"," ",text)
    
    # tokenization: split text into tokens (words) using NLTK
    words = word_tokenize(text)
    
    # stop word removal
    words = [w for w in words if w not in stopwords.words("english")]
    
    # lemmatization
    # reduce words to their root form
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
    # lemmatize verbs by specifying pos
    lemmed = [WordNetLemmatizer().lemmatize(w, pos = 'v') for w in lemmed]
    
    return lemmed



def build_model():
    ''' Build a machine learning pipeline
    Build a pipeline with TFIDF DTM, and a random forest classifier. Grid search on the `use_idf` from tf_idf and `n_estimators` from random forest classifier to find the best model
    

    Args:
        messages_filepath: the path for the messages dataset.
        categories_filepath: the path for the categories dataset.

    Returns:
        cv: the built model

    '''
    # Build a machine learning pipeline
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer = tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

        ])),    

        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # Set up the search grid to find better parameters, which can improve model
    parameters = {
        'features__text_pipeline__tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [50, 100]
    }

    # Initialize GridSearch cross validation object
    cv = GridSearchCV(pipeline, param_grid = parameters)

    return cv



def evaluate_model(model, X_test, Y_test, category_names):
    ''' load data
    Evaluate the model performance of each category target column

    Args:
        model: model object.
        X_test: test feature set.
        Y_test: test target set.
        category_names: target category names

    Returns:
        df: the merged dataset between messages dataset and categories dataset.

    '''

    # Use model to predict
    Y_pred = model.predict(X_test)
    # Turn prediction into DataFrame
    Y_pred = pd.DataFrame(Y_pred, columns = category_names)
    # For each category column, print performance
    for col in category_names:
        print(f'Column Name:{col}\n')
        print(classification_report(Y_test[col],Y_pred[col]))




def save_model(model, model_filepath):
    ''' save the model to a pickle file

    Args:
        model: model object.
        modle_filepath: model output file path

    Returns:
        None

    '''

    joblib.dump(model, model_filepath)



def main():
    ''' load data
    load messages dataset and categories dataset, and output the merged dataset

    Args:
        messages_filepath: the path for the messages dataset.
        categories_filepath: the path for the categories dataset.

    Returns:
        df: the merged dataset between messages dataset and categories dataset.

    '''

    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)   # Split data into train and test sets
        
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