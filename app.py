import streamlit as st
import tweepy
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import nltk
from nltk_download import ensure_nltk_data

# Ensure required corpora are downloaded
ensure_nltk_data()

# App Title
st.set_page_config(page_title="Twitter Sentiment Analyzer", layout="centered")
st.title("📊 Twitter Sentiment Analyzer (Using Tweepy + TextBlob)")

# Twitter API Setup
BEARER_TOKEN = st.secrets["api"]["bearer_token"]
client = tweepy.Client(bearer_token=BEARER_TOKEN)

# User Input
query = st.text_input("Enter a keyword or hashtag (e.g., #AI, Nigeria, Bitcoin):", "#AI")
limit = st.slider("Number of tweets to fetch", 10, 200, 50)

if st.button("Analyze") and query:
    with st.spinner("Fetching tweets..."):
        tweets = client.search_recent_tweets(query=query, max_results=min(limit, 100), tweet_fields=["text", "lang"])
        texts = [tweet.text for tweet in tweets.data if tweet.lang == "en"]

    if not texts:
        st.warning("No English tweets found for this query.")
        st.stop()

    # Sentiment Analysis
    def get_sentiment(text):
        return TextBlob(text).sentiment.polarity

    df = pd.DataFrame(texts, columns=["Tweet"])
    df["Polarity"] = df["Tweet"].apply(get_sentiment)
    df["Sentiment"] = df["Polarity"].apply(
        lambda x: "Positive" if x > 0 else ("Negative" if x < 0 else "Neutral")
    )

    # Summary
    sentiment_counts = df["Sentiment"].value_counts()
    st.subheader("📌 Sentiment Summary")
    st.write(f"🔹 **Positive:** {sentiment_counts.get('Positive', 0)}")
    st.write(f"🔹 **Negative:** {sentiment_counts.get('Negative', 0)}")
    st.write(f"🔹 **Neutral:** {sentiment_counts.get('Neutral', 0)}")

    # Pie Chart
    st.subheader("📈 Sentiment Distribution")
    fig, ax = plt.subplots()
    sentiment_counts.plot.pie(
        autopct="%1.1f%%", colors=["green", "red", "gray"], ax=ax, ylabel=""
    )
    st.pyplot(fig)

    # Sample Tweets
    st.subheader("🗣 Sample Tweets")
    for i, row in df.sample(min(5, len(df))).iterrows():
        st.markdown(f"- *{row['Tweet']}*  \n**Sentiment:** `{row['Sentiment']}`")

    # Insights
    st.subheader("💡 Quick Insight")
    if sentiment_counts.get("Positive", 0) > sentiment_counts.get("Negative", 0):
        st.success("Most users are feeling positive about this topic!")
    elif sentiment_counts.get("Negative", 0) > sentiment_counts.get("Positive", 0):
        st.error("Public sentiment seems largely negative.")
    else:
        st.info("The sentiment is quite neutral.")
