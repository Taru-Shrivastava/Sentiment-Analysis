import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
import string

# Scraping configuration
base_url = "https://www.airlinequality.com/airline-reviews/british-airways"
pages = 5
page_size = 1000

# List to store reviews
reviews = []

# Scraping loop
for i in range(1, pages + 1):
    print(f"Scraping page {i}")
    url = f"{base_url}/page/{i}/?sortby=post_date%3ADesc&pagesize={page_size}"
    response = requests.get(url)
    parsed_content = BeautifulSoup(response.content, 'html.parser')

    for para in parsed_content.find_all("div", {"class": "text_content"}):
        reviews.append(para.get_text().strip())  # Strip whitespace
    
    print(f"   ---> {len(reviews)} total reviews")

# Create DataFrame from reviews
df = pd.DataFrame(reviews, columns=['reviews'])

# Custom function to clean reviews
def clean_text(text):
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    # Remove special characters, numbers, mentions, and hashtags
    text = re.sub(r'\@\w+|\#|\d+', '', text)
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Clean reviews using the custom clean function
df['clean_reviews'] = df['reviews'].apply(clean_text)

# Function to clean "Trip Verified" or "Not Verified" from review text
def clean_verified(review):
    if review.startswith('Trip Verified | '):
        return review[16:]
    elif review.startswith('Not Verified | '):
        return review[15:]
    return review

df['clean_reviews'] = df['clean_reviews'].apply(clean_verified)

# Function to remove punctuation
def punctuation_removal(text):
    return text.translate(str.maketrans('', '', string.punctuation))

df['clean_reviews'] = df['clean_reviews'].apply(punctuation_removal)

# Filter out stopwords
stop_words = set(stopwords.words('english'))

def filter_words(text):
    word_tokens = word_tokenize(text)
    return [w for w in word_tokens if w not in stop_words and w.isalpha()]

df['filtered_reviews'] = df['clean_reviews'].apply(filter_words)

# Function to plot word cloud
def plot_wordcloud(review, title, max_words):
    text = " ".join(review)
    word_cloud = WordCloud(
        background_color="white",
        stopwords=stop_words,
        max_words=max_words,
        width=800,
        height=800
    ).generate(text)

    plt.figure(figsize=(10, 10))
    plt.imshow(word_cloud, interpolation="bilinear")
    plt.axis('off')
    plt.title(title)
    plt.show()

# Plot word cloud for the reviews
all_reviews_text = df['filtered_reviews'].apply(lambda x: ' '.join(x)).tolist()
plot_wordcloud(all_reviews_text, "British Airways Reviews Word Cloud", max_words=100)

# Lemmatization
lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in text]

df['lemmatized_reviews'] = df['filtered_reviews'].apply(lemmatize_text)

# Sentiment analysis
def sentiment_analyzer(review):
    score = TextBlob(review).sentiment.polarity
    if score > 0:
        return "positive"
    elif score < 0:
        return "negative"
    else:
        return "neutral"

# Apply sentiment analysis to clean reviews
df['sentiment'] = df['clean_reviews'].apply(sentiment_analyzer)

# Plot sentiment analysis results
sns.countplot(data=df, x='sentiment')
plt.title('Sentiment Analysis of British Airways Reviews')
plt.show()
