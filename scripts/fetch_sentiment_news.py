import requests
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
import time

#CONFIG 
API_KEY = "8903334355ec4296bd0c5a1802488fc5"  #API Key
QUERY = "Reliance Industries"
FROM_DATE = "2020-01-01"
TO_DATE = "2024-12-31"
PAGE_SIZE = 100  # Max per NewsAPI
MAX_PAGES = 5    

#INITIAL SETUP
all_articles = []
analyzer = SentimentIntensityAnalyzer()

#PAGINATED REQUEST
for page in range(1, MAX_PAGES + 1):
    print(f"Fetching page {page}...")
    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={QUERY}&"
        f"from={FROM_DATE}&"
        f"to={TO_DATE}&"
        f"language=en&"
        f"sortBy=publishedAt&"
        f"pageSize={PAGE_SIZE}&"
        f"page={page}&"
        f"apiKey={API_KEY}"
    )
    
    response = requests.get(url)
    if response.status_code != 200:
        print("Error:", response.status_code, response.text)
        break

    data = response.json()
    articles = data.get("articles", [])
    if not articles:
        break

    for art in articles:
        date = art['publishedAt'][:10]
        headline = art['title']
        score = analyzer.polarity_scores(headline)['compound']
        all_articles.append((date, headline, score))
    
    time.sleep(1)  #API rate limit

#CREATE DATAFRAME
df_news = pd.DataFrame(all_articles, columns=['Date', 'Headline', 'Sentiment'])
df_news['Date'] = pd.to_datetime(df_news['Date'])

#DAILY AVERAGE SENTIMENT 
daily_sentiment = df_news.groupby('Date')['Sentiment'].mean().reset_index()

#SAVE TO CSV
df_news.to_csv("reliance_news_headlines.csv", index=False)
daily_sentiment.to_csv("reliance_daily_sentiment.csv", index=False)

print("✅ Done! Saved:")
print("- All headlines to `reliance_news_headlines.csv`")
print("- Daily sentiment to `reliance_daily_sentiment.csv`")
