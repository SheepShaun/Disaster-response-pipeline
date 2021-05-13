import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
	''' load data
	load messages dataset and categories dataset, and output the merged dataset

	Args:
		messages_filepath: the path for the messages dataset.
		categories_filepath: the path for the categories dataset.

	Returns:
		df: the merged dataset between messages dataset and categories dataset.

	'''

	# load message dataset
	messages = pd.read_csv(messages_filepath)

	# load categories dataset
	categories = pd.read_csv(categories_filepath)

	# merge datasets
	df = pd.merge(messages, categories, on = 'id')

	return df


def clean_data(df):
	''' clean the merged dataset
	1. Split the categories column into separate category columns;
	2. Convert category values to just numbers 0 or 1;
	3. Replace the categories column in df with new category columns;
	4. Remove duplicates.

	Args:
		df: the merged dataset.

	Returns:
		df: the cleaned dataset.

	'''
	# Split 'categories' column into separate category columns    
	categories = df['categories'].str.split(';', 36, expand = True)   # create a dataframe of the 36 individual category columns   
	
	row = categories.iloc[0]    # select the first row of the categories dataframe    
	
	category_colnames = row.str.split('-').apply(lambda x: x[0])  # use this row to extract a list of new column names for categories  
	
	categories.columns = category_colnames  # rename the columns of 'categories'

	# Convert category values to just numbers
	for column in categories:
	  
		categories[column] = categories[column].str.split('-').apply(lambda x:x[1])    # set each value to be the last character of the string
		# or   # categories[column] = categories[column].astype(str).str[-1]
			
		categories[column] = categories[column].astype(int)  # convert column from string to numeric

	# Replace categories column in df with new category columns
	df = df.drop(columns = ['categories'])   # drop the original categories column from `df`

	df = pd.concat([df, categories], axis = 1)  # concatenate the original dataframe with the new `categories` dataframe

	# Remove duplicates
	df = df.drop_duplicates()	# drop duplicates

	# For the values for each category, remove the rows with value equals to 2 and only keep rows with binary values (i.e either 0 or 1)
	categoryName_list = df.columns.tolist()[4:41]   # a list for 36 category names

	for column in categoryName_list:
		df = df[(df[column] == 0) & (df[columm] == 1)] # only keep rows with binary values, for each category

	return df


def save_data(df, database_filename):
	''' save the clean dataset into an sqlite database

	Args:
		df: the merged dataset.
		database_filename: the file name of the sqlite database

	Returns:
		none

	'''

	engine = create_engine('sqlite:///' + database_filename)
	df.to_sql('Disaster_Response', engine, index = False, if_exists="replace")


def main():
	''' run the main function to process the data from two datasets

	Args:
		sys.argv: inputs from command line

	Returns:
		none

	'''

	if len(sys.argv) == 4:

		messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

		print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
			  .format(messages_filepath, categories_filepath))
		df = load_data(messages_filepath, categories_filepath)

		print('Cleaning data...')
		df = clean_data(df)
		
		print('Saving data...\n    DATABASE: {}'.format(database_filepath))
		save_data(df, database_filepath)
		
		print('Cleaned data saved to database!')
	
	else:
		print('Please provide the filepaths of the messages and categories '\
			  'datasets as the first and second argument respectively, as '\
			  'well as the filepath of the database to save the cleaned data '\
			  'to as the third argument. \n\nExample: python process_data.py '\
			  'disaster_messages.csv disaster_categories.csv '\
			  'DisasterResponse.db')


if __name__ == '__main__':
	main()