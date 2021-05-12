# Disaster-response-pipeline-project
**Goal:** <br/>
This project aims to create pipelines to process disaster messages data and build a model for an API in order to classify these disaster messages.

**This project includes three parts:** <br/>
1) ETL pipeline <br/>
In a Python script, process_data.py, the data cleaning pipeline, ETL pipeline, was created to: 
<br/> - Load two datasets: disaster_messages.csv and disaster_categories.csv;
<br/> - Merges the two datasets;
<br/> - Splits the categories column into separate;
<br/> - Cleans the data (clearly named columns, converts values to binary, and drops duplicates);
<br/> - Stores it in a SQLite database (DisasterResponse.db).

2) ML pipeline  <br/>
A Python script, train_classifier.py, was created to: 
<br/> - Loads data from the SQLite database;
<br/> - Splits the dataset into the training set and the test set;
<br/> - Builds a text processing (using a custom tokenize function using nltk to case normalize, lemmatize, and tokenize text) and machine learning pipeline;
<br/> - Trains and tunes a model using GridSearchCV;
<br/> - Outputs results on the test set;
<br/> - Exports the final model as a pickle file (classifier.pkl).

3) Flask Web App <br/>
The app was built to display visualizations of the data and provide a query to input a new message and classify it to serveral categories.
In the web app, the data, which was extracted from the SQLite database, was visualized using Plotly.

**Instructions about how to run the Python scripts and web app:** <br/>
1. Run the following commands in the project's root directory to set up the database and model.

   - To run ETL pipeline (process_data.py) that cleans data and stores in database
     `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
   - To run ML pipeline (train_classifier.py) that trains classifier and saves
     `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run the web app.
   `python run.py`

   Go to http://127.0.0.1:3001/ to see the web page.
