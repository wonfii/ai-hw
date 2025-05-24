import pandas as pd
from textblob import TextBlob

df = pd.read_csv('product_reviews.csv')

df['Polarity'] = df['Review'].apply(lambda x: TextBlob(x).sentiment.polarity)

def classify_sentiment(polarity):
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

df['Sentiment'] = df['Polarity'].apply(classify_sentiment)

sentiment_counts = df['Sentiment'].value_counts()

print(sentiment_counts)