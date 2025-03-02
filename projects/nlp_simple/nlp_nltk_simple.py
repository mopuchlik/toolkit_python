# %%

import nltk
import string
import random
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy
from sklearn.model_selection import train_test_split

# %%
text = "NLTK makes Natural Language Processing easy! AI is evolving rapidly, and chatbots are improving day by day."

# %%
# 1. Tokenization
words = word_tokenize(text)

# %%
# 2. Remove Stopwords & Punctuation
stop_words = set(stopwords.words("english"))
filtered_words = [
    word.lower()
    for word in words
    if word.lower() not in stop_words and word not in string.punctuation
]

# %%
# 3. Stemming & Lemmatization
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

stemmed_words = [stemmer.stem(word) for word in filtered_words]
lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]

# %%
# 4. POS Tagging
pos_tags = pos_tag(lemmatized_words)

# %%
# 5. Sentiment Analysis
nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()
sentiment_scores = sia.polarity_scores(text)
sentiment = "Positive" if sentiment_scores["compound"] > 0 else "Negative"

# %%

print("Original Text:", text)
print("Tokenized Words:", words)
print("Filtered Words:", filtered_words)
print("Stemmed Words:", stemmed_words)
print("Lemmatized Words:", lemmatized_words)
print("POS Tags:", pos_tags)
print("Sentiment:", sentiment)


# %%

# Sample labeled dataset (Technology vs. Non-Technology)
dataset = [
    ("AI and NLP are transforming industries.", "tech"),
    ("New smartphone releases are exciting.", "tech"),
    ("The economy is showing signs of recovery.", "non-tech"),
    ("The government passed a new policy today.", "non-tech"),
    ("Chatbots are improving with deep learning.", "tech"),
    ("Climate change is a global concern.", "non-tech"),
]


# Preprocessing function
def preprocess(text):
    words = word_tokenize(text.lower())
    words = [
        word
        for word in words
        if word not in stop_words and word not in string.punctuation
    ]
    return {word: True for word in words}  # Convert to feature dict for NLTK model


# Prepare dataset for classification
featuresets = [(preprocess(text), category) for (text, category) in dataset]

# Train-test split
train_set, test_set = train_test_split(featuresets, test_size=0.2, random_state=42)

# Train Naïve Bayes Classifier
classifier = NaiveBayesClassifier.train(train_set)

# Evaluate Accuracy
print("Model Accuracy:", accuracy(classifier, test_set))

# %%
new_text = "AI is making chatbots smarter!"
processed_text = preprocess(new_text)

# Predict category
predicted_category = classifier.classify(processed_text)
print(f"Predicted Category: {predicted_category}")

# %%
