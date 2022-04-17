# Disaster Response Pipeline Project

### Table of Contents
1. Summary(#summary)
2. [Installation](#installation)
3. [File Descriptions](#files)
4. [Instructions](#instructions)

## Summary <a name="summary"></a>
This project analyzes disaster data from Figure Eight to build a model for an API that classifies disaster messages.
The data set contains real messages that were sent during disaster events.
Goal of this project is to create a machine learning pipeline to categorize these events so that you can send the messages to an appropriate disaster relief agency.

## Installation <a name="installation"></a>

To install the required packages run the code below
 'pip install -r requirements.txt'


### File Descriptions <a name="files"></a>

1. ETL Pipeline
There is 1 python file 'data/process_data.py' with the code used to read the data and store them in a SQLite database.

2. ML Pipeline
There is 1 python file 'models/train_classifier.py' that trains the classifier and saves the model.

3. Flask Web App
There is 1 python file 'models/features.py' with the class to generate additional feature for the classification.


### Instructions <a name="instructions"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python app/run.py`

3. Go to http://0.0.0.0:3001/
   If you are unable to connect to http://0.0.0.0:3001/ follow the steps below.
  - Open a terminal and type env|grep WORK this will give you the spaceid (it will start with view*** and some characters after that)
  - Now open your browser window and type https://spaceid-3001.udacity-student-workspaces.com. Replace the whole 'spaceid' with your space id that you got in the step 2
  - Press enter and the app should now run for you
