Written by Ankita Patra

# Fake News Detection

This project implements a machine learning-based system for detecting fake news. The system utilizes a Naive Bayes classifier trained on textual data to classify articles as either Fake News or Real News. Additionally, the project addresses the issue of imbalanced data using resampling techniques such as SMOTE or RandomOverSampler.

## Table of Contents

- [Features]
- [Requirements]
- [Installation]
- [Usage]
- [Run]
- [output]

## Features

- Preprocessing of text data, including stopword removal and punctuation cleaning.
- TF-IDF vectorization to convert text into numerical representations.
- Naive Bayes classifier for predicting fake or real news.
- Balancing imbalanced datasets using SMOTE or RandomOverSampler.
- Visual representation of the class distribution in the dataset.
- Ability to predict the classification of new articles provided in a text file.


## Requirements

To run the project, ensure you have the following dependencies installed:
Python 3.8
pandas
scikit-learn
imbalanced-learn (for SMOTE/oversampling)
nltk
matplotlib


You can install the required packages using pip:

## Installation
- pip3 install pandas scikit-learn nltk
- pip3 install matplotlib
- you can install all dependncies using following command:
        pip install -r requirements.txt
    - requirements.txt :(optional)
                        pandas
                        scikit-learn
                        imbalanced-learn
                        nltk
                        matplotlib




## files 
- fake_news_detection.py : The main Python script that runs the fake news detection system.
- README.md : Documentation for the project.
- fake_news_detection.txt : he input file containing news articles for classification.
- requirements.txt (optional) : Dependencies list 


## Usage

- Text Preprocessing and Model Training:

* The script reads a sample dataset of news articles with labels indicating fake (1) or real (0) news.
* The text data undergoes preprocessing, which includes converting text to lowercase, removing stopwords, and stripping punctuation.
* The preprocessed text is transformed into numerical features using TF-IDF vectorization.
* The dataset is split into training and test sets. If the data is imbalanced, SMOTE or RandomOverSampler is applied.
* The Naive Bayes classifier is trained on the processed data and evaluated on the test set.

- Classifying New Articles:

* The script can read new articles from the fake_news_detection.txt file, preprocess the text, and classify each article as Fake News or Real News based on the trained model.
* Ensure the fake_news_detection.txt file is present in the project directory, with one article per line.

## Run:

    python3 fake_news_detection.py

## Output:

* A bar chart showing the distribution of fake vs. real news in the sample dataset.
* Classification metrics including accuracy, precision, recall, and F1-score.
* Predictions for new articles, if provided.


