# Sentiment-Analysis
This Python script is designed to scrape customer reviews of British Airways from a specified website, clean the data, and perform sentiment analysis. Here's a step-by-step breakdown of its functionality:

1. **Web Scraping:**
   - The script uses `requests` to download HTML content from several pages of British Airways customer reviews.
   - `BeautifulSoup` is used to parse the HTML and extract reviews from specific elements on each page.
   - The reviews are stored in a list and then converted into a Pandas DataFrame.

2. **Data Cleaning:**
   - The `clean_text` function removes URLs, special characters, numbers, mentions, hashtags, and extra whitespace from the review text.
   - Another function, `clean_verified`, removes the "Trip Verified" or "Not Verified" labels that precede some reviews.
   - Punctuation is removed from the cleaned reviews to make text processing easier.

3. **Stopword Filtering:**
   - Using the `nltk` library, stopwords (common words like "the", "and", etc.) are filtered out, leaving only meaningful words.
   - Word tokens are extracted and filtered, keeping only alphabetic tokens.

4. **Word Cloud Generation:**
   - A word cloud is generated to visually represent the most frequent words found in the reviews.
   - The cloud highlights terms based on frequency, showing larger words for more frequently occurring terms.

5. **Lemmatization:**
   - Lemmatization is performed on the filtered reviews using NLTK's WordNetLemmatizer. This step reduces words to their base form (e.g., "running" becomes "run").
   - The lemmatized words are stored in a new column for further processing.

6. **Sentiment Analysis:**
   - Sentiment analysis is performed using the `TextBlob` library to determine if a review's sentiment is positive, negative, or neutral.
   - The sentiment score is computed based on the polarity of each review and categorized accordingly.

7. **Visualization:**
   - A Seaborn bar chart is used to visualize the distribution of sentiments (positive, negative, neutral) among the reviews.

This script showcases the entire process of scraping text data, cleaning it, performing natural language processing, and generating meaningful insights via sentiment analysis and visualizations.
