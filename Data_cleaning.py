# Prompt : Write Python Script Code where if I feed any sort of unstructured or structured data. 
# it will automate its data Cleaning process, which will help to prepare Machine learning models. 
# Make sure to add all sort of Data Cleaning Adds in this Script.

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')

def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

def clean_data(data):
    # Remove duplicate rows
    data = data.drop_duplicates()

    # Handling missing values
    data.dropna(subset=['target_column'], inplace=True)  # Drop rows with missing target
    data.dropna(thresh=len(data) * 0.7, axis=1, inplace=True)  # Drop columns with >70% missing data

    # Impute missing values for numerical columns
    numerical_columns = data.select_dtypes(include=[np.number]).columns
    imputer = SimpleImputer(strategy='mean')
    data[numerical_columns] = imputer.fit_transform(data[numerical_columns])

    # Standardize numerical features
    scaler = StandardScaler()
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

    # Label encode categorical variables
    label_encoder = LabelEncoder()
    categorical_columns = data.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        data[column] = label_encoder.fit_transform(data[column].astype(str))

    # Clean and preprocess text data
    text_columns = data.select_dtypes(include=['object']).columns
    for column in text_columns:
        data[column] = data[column].apply(clean_text)

        # Convert text to numerical features using TF-IDF
        tfidf_vectorizer = TfidfVectorizer(max_features=1000)
        tfidf_matrix = tfidf_vectorizer.fit_transform(data[column])
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
        data = pd.concat([data, tfidf_df], axis=1)
        data.drop(column, axis=1, inplace=True)

    return data

# Example usage
raw_data = pd.read_csv('your_data.csv')  # Replace with your data source
cleaned_data = clean_data(raw_data)

# Now you can use the cleaned_data to train your machine learning models


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy import stats

def load_dataset(file_path):
    data = pd.read_csv(file_path)
    return data

def explore_data(data):
    print(data.info())
    print(data.head())
    print(data.isnull().sum())

def handle_missing_values(data):
    # Drop rows with any missing values
    data_cleaned = data.dropna()
    
    # Fill missing values with mean or median
    data['Age'].fillna(data['Age'].mean(), inplace=True)
    data['Income'].fillna(data['Income'].median(), inplace=True)

    return data_cleaned

def convert_data_types(data):
    data['Age'] = data['Age'].astype(int)

    return data

def remove_duplicates(data):
    data.drop_duplicates(inplace=True)

def handle_outliers(data):
    # Identify and remove outliers using Z-score
    z_scores = np.abs(stats.zscore(data[['Age', 'Income']]))
    data_cleaned = data[(z_scores < 3).all(axis=1)]

    return data_cleaned

def feature_engineering(data):
    data['BMI'] = data['Weight'] / (data['Height'] ** 2)

    return data

def normalize_features(data):
    scaler = MinMaxScaler()
    data[['Age', 'Income', 'Height', 'Weight']] = scaler.fit_transform(data[['Age', 'Income', 'Height', 'Weight']])

    return data

def save_cleaned_data(data, output_file):
    data.to_csv(output_file, index=False)

if __name__ == "__main__":
    # Step 1: Load the dataset
    file_path = 'your_dataset.csv'
    dataset = load_dataset(file_path)

    # Step 2: Explore the data
    explore_data(dataset)

    # Step 3: Handle missing values
    dataset = handle_missing_values(dataset)

    # Step 4: Convert data types
    dataset = convert_data_types(dataset)

    # Step 5: Remove duplicates
    remove_duplicates(dataset)

    # Step 6: Handle outliers
    dataset = handle_outliers(dataset)

    # Step 7: Feature engineering
    dataset = feature_engineering(dataset)

    # Step 8: Normalize features
    dataset = normalize_features(dataset)

    # Step 9: Save the cleaned data
    output_file = 'cleaned_dataset.csv'
    save_cleaned_data(dataset, output_file)