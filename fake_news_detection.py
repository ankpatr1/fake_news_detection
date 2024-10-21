# Written By Ankita Patra
# Project is all about Fake news detection using an ML-based system.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import RandomOverSampler
import nltk
from nltk.corpus import stopwords
import string
import matplotlib.pyplot as plt

# Download NLTK stopwords 
nltk.download('stopwords')

# Sample dataset containing text articles and labels
data = {
    "content": [
        "Reports suggest the election results were manipulated.",
        "Researchers have identified a new celestial body.",
        "There are rumors about a celebrity's new relationship.",
        "Some claim the Earth is flat based on certain beliefs.",
        "Studies indicate that chocolate might be beneficial for health.",
        "Conspiracy theories argue that the moon landing was a hoax."
    ],
    "label": [1, 0, 1, 1, 0, 1]  # 1 indicates fake news, 0 indicates real news
}

# Create a DataFrame using the defined dataset
df = pd.DataFrame(data)

# Check class distribution
class_counts = df['label'].value_counts()
print("Class Distribution:\n", class_counts)

# Visualize class distribution
class_counts.plot(kind='bar', title='Class Distribution (0: Real News, 1: Fake News)')
plt.xlabel('Label')
plt.ylabel('Count')
plt.show()

# Function to clean and process text
def clean_and_process(text):
    stop_words = set(stopwords.words("english"))
    text = text.lower()  # Convert text to lowercase
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    text = " ".join(word for word in text.split() if word not in stop_words)  # Filter out stop words
    return text

# Clean the text data
df["content"] = df["content"].apply(clean_and_process)

# Split the data into training and testing sets with stratification
X_train, X_test, y_train, y_test = train_test_split(
    df["content"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

# Convert the text data into TF-IDF features
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Check if the dataset is too small for SMOTE
min_class_count = min(y_train.value_counts())

if min_class_count > 1:
    # If the smallest class has more than 1 sample, use SMOTE (alternative method)
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42, k_neighbors=1)
    X_train_res, y_train_res = smote.fit_resample(X_train_tfidf, y_train)
    print("Using SMOTE for oversampling.")
else:
    # Use RandomOverSampler instead for very small datasets
    ros = RandomOverSampler(random_state=42)
    X_train_res, y_train_res = ros.fit_resample(X_train_tfidf, y_train)
    print("Using RandomOverSampler due to small class size.")

# Initialize and train the Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train_res, y_train_res)

# Make predictions on the test set
predictions = classifier.predict(X_test_tfidf)

# Evaluate the classifier's performance
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions, zero_division=0)

print(f"Model Accuracy: {accuracy:.2f}")
print("Classification Report:\n", report)

# Load new articles from a text file
try:
    with open("fake_news_detection.txt", "r") as file:
        new_articles = file.readlines()
except FileNotFoundError:
    print("The file 'fake_news_detection.txt' was not found. Please ensure it exists in the project directory.")
    new_articles = []

# Clean the articles read from the file
new_articles = [article.strip() for article in new_articles if article.strip()]  # Remove whitespace and empty lines

# Check if there are new articles to process
if new_articles:
    # Process and vectorize the new articles
    processed_articles = [clean_and_process(article) for article in new_articles]
    article_vectors = vectorizer.transform(processed_articles)

    # Predict the labels for the new articles
    article_predictions = classifier.predict(article_vectors)

    # Output the predictions
    for article, prediction in zip(new_articles, article_predictions):
        print(f'Article: "{article}" -> Classification: {"Fake News" if prediction == 1 else "Real News"}')
else:
    print("No new articles to classify.")
