# Import Python Libraries
import pickle
import string
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# Importing DataSet
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

datasets = pd.read_csv("E:\Data_World\Care_Medico.csv")
datasets.head(9)

datasets.info()

# Before
# checking for null values
datasets.isnull().sum()

# Dropping null values
datasets = datasets.dropna(axis=0)

# After
# checking for null values
datasets.isnull().sum()

# Changing the "Unnamed: 0" column to uniqueId as it represents the unique id of the drugs
datasets = datasets.drop('Unnamed: 0', axis='columns')


# Using cleaner function to check out irrelevent words in the dataset
def cleaner(x):
    return [a for a in (''.join([a for a in x if a not in string.punctuation])).lower().split()]


# Decision Tree Classifier

Pipe= Pipeline([
    ('bow', CountVectorizer(analyzer=cleaner)),
    ('tfidf', TfidfTransformer()),
    ('classifier', DecisionTreeClassifier())
])

ab = Pipe.fit(datasets['condition'], datasets['drugName'])

Pipe.predict(["Cough & Headache"])[0]


# Dump the trained decision tree classifier with Pickle
Classifier_filename = 'Chain_medico_care.pkl'
# Open the file to save as pkl file
model_pkl = open(Classifier_filename, 'wb')
pickle.dump(ab, model_pkl)
# Close the pickle instances
model_pkl.close()