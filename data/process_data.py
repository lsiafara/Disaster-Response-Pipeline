import sys

import pandas as pd
import sqlalchemy


def load_data(messages_filepath, categories_filepath): 
    '''
    INPUT 
        messages_filepath - a string with the path to the messages csv file
        categories_filepath - a string with the path to the categories csv file
        
    OUTPUT
        df - a dataframe with the merged messages and categories columns
    '''
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, how= "inner", on='id')
    return df


def clean_data(df):
    '''
    INPUT 
        df - a dataframe with the merged messages and categories columns
        
    OUTPUT
        df - the df dataframe with the cleaned data
    '''

    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';', expand= True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[:1]
    
    # use this row to extract a list of new column names for categories.
    category_colnames = row.apply(lambda x: x.str.slice(stop=-2)).values.tolist()
    categories.columns = category_colnames[0]
    
    # iterate through the category columns in df to keep only the last character of each string (the 1 or 0).
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.slice(start=-1)
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    # drop old categories column
    df.drop(columns='categories', axis=0, inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1, ignore_index=False)
    
    # remove duplicates
    df = df[~df.duplicated()]

    # replace 2s with 1a in the column 'related'
    # for explanation how to treat 2s see this post:
    # https://knowledge.udacity.com/questions/136791
    df['related'] = df['related'].replace(2,1)

    return df


def save_data(df, database_filepath):
    '''
    Stores a dataframe into an sql database

    INPUT 
        df - a dataframe with the data to store in an sql database
        database_filepath - the path to the sql database
    '''

    engine = sqlalchemy.create_engine('sqlite:///' + database_filepath)
    df.to_sql('DisasterResponse', engine, if_exists='replace', index=False)  


def main():
    '''
    Loads, cleans, and saves the data into an sql database
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
