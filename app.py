import streamlit as st
import tweepy
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import nltk
from nltk_download import ensure_nltk_data
import time
from datetime import datetime, timedelta

# Ensure required corpora are downloaded
ensure_nltk_data()

# App Title
st.set_page_config(page_title="Twitter Sentiment Analyzer", layout="centered")
st.title("📊 Twitter Sentiment Analyzer (Using Tweepy + TextBlob)")

# Twitter API Setup
try:
    BEARER_TOKEN = st.secrets["api"]["bearer_token"]
    client = tweepy.Client(bearer_token=BEARER_TOKEN, wait_on_rate_limit=True)
except Exception as e:
    st.error("❌ Error setting up Twitter API. Please check your bearer token.")
    st.stop()

# User Input
query = st.text_input("Enter a keyword or hashtag (e.g., #AI, Nigeria, Bitcoin):", "#AI")
limit = st.slider("Number of tweets to fetch", 10, 100, 50)  # Reduced max to 100

# Add rate limit info
st.info("ℹ️ **Rate Limit Info**: Twitter API allows 300 requests per 15 minutes. Each request fetches up to 100 tweets.")

if st.button("Analyze") and query:
    try:
        with st.spinner("Fetching tweets... This may take a moment due to rate limiting."):
            # Add some basic query validation
            processed_query = query.strip()
            if not processed_query.startswith('#') and not processed_query.startswith('@'):
                processed_query = f'"{processed_query}"'  # Quote non-hashtag queries
            
            # Fetch tweets with error handling
            tweets_response = client.search_recent_tweets(
                query=f"{processed_query} lang:en -is:retweet",  # Only English, no retweets
                max_results=min(limit, 100),
                tweet_fields=["text", "lang", "created_at", "public_metrics"]
            )
            
            if not tweets_response.data:
                st.warning("No tweets found for this query. Try a different keyword or check if it's trending.")
                st.stop()
            
            # Process tweets
            texts = [tweet.text for tweet in tweets_response.data]
            
    except tweepy.errors.TooManyRequests:
        st.error("🚫 **Rate limit exceeded!** Please wait 15 minutes before trying again.")
        st.info("💡 **Tip**: The Twitter API has a limit of 300 requests per 15-minute window.")
        st.stop()
        
    except tweepy.errors.Unauthorized:
        st.error("🔐 **Authentication failed!** Please check your Bearer Token.")
        st.stop()
        
    except tweepy.errors.BadRequest as e:
        st.error(f"❌ **Bad request**: {str(e)}")
        st.info("💡 Try simplifying your search query or removing special characters.")
        st.stop()
        
    except Exception as e:
        st.error(f"❌ **Unexpected error**: {str(e)}")
        st.stop()

    if not texts:
        st.warning("No tweets found after filtering. Try a different query.")
        st.stop()

    # Sentiment Analysis
    @st.cache_data
    def get_sentiment(text):
        return TextBlob(text).sentiment.polarity

    with st.spinner("Analyzing sentiment..."):
        df = pd.DataFrame(texts, columns=["Tweet"])
        df["Polarity"] = df["Tweet"].apply(get_sentiment)
        df["Sentiment"] = df["Polarity"].apply(
            lambda x: "Positive" if x > 0.1 else ("Negative" if x < -0.1 else "Neutral")
        )

    # Summary with better formatting
    sentiment_counts = df["Sentiment"].value_counts()
    total_tweets = len(df)
    
    st.subheader(f"📌 Sentiment Summary ({total_tweets} tweets analyzed)")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        pos_count = sentiment_counts.get('Positive', 0)
        pos_pct = (pos_count / total_tweets) * 100 if total_tweets > 0 else 0
        st.metric("🟢 Positive", f"{pos_count}", f"{pos_pct:.1f}%")
    
    with col2:
        neg_count = sentiment_counts.get('Negative', 0)
        neg_pct = (neg_count / total_tweets) * 100 if total_tweets > 0 else 0
        st.metric("🔴 Negative", f"{neg_count}", f"{neg_pct:.1f}%")
    
    with col3:
        neu_count = sentiment_counts.get('Neutral', 0)
        neu_pct = (neu_count / total_tweets) * 100 if total_tweets > 0 else 0
        st.metric("⚪ Neutral", f"{neu_count}", f"{neu_pct:.1f}%")

    # Pie Chart with better colors
    st.subheader("📈 Sentiment Distribution")
    if len(sentiment_counts) > 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = {'Positive': '#2E8B57', 'Negative': '#DC143C', 'Neutral': '#708090'}
        plot_colors = [colors.get(label, '#708090') for label in sentiment_counts.index]
        
        sentiment_counts.plot.pie(
            autopct="%1.1f%%", 
            colors=plot_colors, 
            ax=ax, 
            ylabel="",
            startangle=90
        )
        plt.title("Sentiment Distribution")
        st.pyplot(fig)

    # Sample Tweets by Sentiment
    st.subheader("🗣 Sample Tweets by Sentiment")
    
    for sentiment in ['Positive', 'Negative', 'Neutral']:
        sentiment_tweets = df[df['Sentiment'] == sentiment]
        if not sentiment_tweets.empty:
            st.markdown(f"### {sentiment} Tweets")
            sample_tweets = sentiment_tweets.sample(min(3, len(sentiment_tweets)))
            for _, row in sample_tweets.iterrows():
                st.markdown(f"- *{row['Tweet'][:200]}{'...' if len(row['Tweet']) > 200 else ''}*")
            st.markdown("---")

    # Enhanced Insights
    st.subheader("💡 Sentiment Insights")
    avg_polarity = df['Polarity'].mean()
    
    if pos_count > neg_count and pos_count > neu_count:
        st.success(f"🎉 **Predominantly Positive** - {pos_pct:.1f}% of tweets show positive sentiment!")
    elif neg_count > pos_count and neg_count > neu_count:
        st.error(f"😟 **Predominantly Negative** - {neg_pct:.1f}% of tweets show negative sentiment.")
    else:
        st.info(f"😐 **Mixed/Neutral Sentiment** - Public opinion appears divided or neutral.")
    
    st.write(f"**Average Sentiment Score**: {avg_polarity:.3f} (Range: -1.0 to +1.0)")
    
    # Download option
    st.subheader("💾 Download Results")
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"sentiment_analysis_{query.replace('#', '').replace('@', '')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

# Add footer with tips
st.markdown("---")
st.markdown("""
### 💡 Tips for Better Results:
- Use specific hashtags or keywords (e.g., #Python, #AI, Bitcoin)
- Popular topics will have more recent tweets
- The API only searches tweets from the last 7 days
- Try different keywords if you get no results
""")

# Debug info (only show if there's an error)
if st.checkbox("Show Debug Info"):
    st.code(f"""
Debug Information:
- Query processed: {query}
- Timestamp: {datetime.now()}
- Rate limits reset every 15 minutes
    """)
