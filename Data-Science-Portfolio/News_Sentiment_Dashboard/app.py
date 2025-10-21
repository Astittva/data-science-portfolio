# 📰 Live News Sentiment Dashboard (Free)
# ---------------------------------------
# Features:
# ✅ Fetches headlines from Google News RSS (free)
# ✅ Sentiment analysis using TextBlob
# ✅ Sentiment distribution chart
# ✅ Word Cloud (all / by sentiment)
# ✅ Time-Series Sentiment Trend
# ✅ CSV Download
# ✅ Professional footer & disclaimer

import streamlit as st
import pandas as pd
import plotly.express as px
from textblob import TextBlob
import feedparser
import re, urllib.parse, requests
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# --- Page setup ---
st.set_page_config(page_title="Live News Sentiment Dashboard", page_icon="📰", layout="wide")
st.title("📰 Live News Sentiment Dashboard")
st.caption("Analyze the tone of recent news headlines — powered by Google News RSS, free and live!")

# --- User inputs ---
topic = st.text_input("Enter a topic (e.g., Cricket, AI, Politics):")
limit = st.slider("Select number of news articles to analyze:", 10, 200, 80, step=10)
split_clouds = st.checkbox("Show separate word clouds for Positive / Negative", value=False)
btn = st.button("🚀 Fetch & Analyze News")

# --- Helper functions ---
def google_news_rss_url(q, lang="en", country="IN"):
    q_enc = urllib.parse.quote(q)
    return f"https://news.google.com/rss/search?q={q_enc}&hl={lang}-{country}&gl={country}&ceid={country}:{lang}"

def clean_text(text):
    text = re.sub(r"<[^>]+>", " ", str(text))
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^A-Za-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

@st.cache_data(show_spinner=False)
def fetch_news(query, n):
    """Fetch headlines from Google News RSS."""
    url = google_news_rss_url(query)
    _ = requests.get(url, timeout=10)
    feed = feedparser.parse(url)
    rows = []
    for entry in feed.entries[:n]:
        rows.append([
            entry.get("published", ""),
            clean_text(entry.get("title", "")),
            clean_text(entry.get("summary", "")),
            entry.get("link", "")
        ])
    return pd.DataFrame(rows, columns=["Published", "Title", "Summary", "Link"])

def analyze_sentiment(df):
    """Apply TextBlob sentiment analysis."""
    df = df.copy()
    df["Text"] = (df["Title"].fillna("") + ". " + df["Summary"].fillna(""))
    df["Polarity"] = df["Text"].apply(lambda x: TextBlob(x).sentiment.polarity)
    df["Sentiment"] = df["Polarity"].apply(lambda x: "Positive" if x > 0 else ("Negative" if x < 0 else "Neutral"))
    df["Published"] = pd.to_datetime(df["Published"], errors="coerce")
    return df

def make_wordcloud(text: str, extra_stopwords=None, max_words=200):
    stops = set(STOPWORDS)
    stops.update({"news", "say", "says", "report", "reports", "today", "will"})
    if extra_stopwords:
        stops.update(extra_stopwords)
    wc = WordCloud(width=1200, height=500, background_color="white",
                   stopwords=stops, max_words=max_words).generate(text)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    return fig

def text_join(series: pd.Series) -> str:
    return " ".join([t for t in series.astype(str).tolist() if t])

# --- Main logic ---
if btn:
    if not topic:
        st.warning("⚠️ Please enter a topic before analyzing.")
        st.stop()

    with st.spinner("Fetching latest news..."):
        df = fetch_news(topic, limit)

    if df.empty:
        st.error("No news articles found. Try another topic.")
        st.stop()

    df = analyze_sentiment(df)

    # ---- Sentiment Distribution ----
    st.subheader(f"📊 Sentiment Analysis for '{topic}' — {len(df)} articles")
    fig = px.histogram(df, x="Sentiment", color="Sentiment", title="Sentiment Distribution")
    st.plotly_chart(fig, use_container_width=True)

    # ---- Time-Series Trend ----
    st.subheader("📈 Sentiment Trend Over Time")
    df_trend = df.groupby(["Published", "Sentiment"]).size().reset_index(name="Count")
    if not df_trend.empty and df_trend["Published"].notna().any():
        fig_trend = px.line(df_trend, x="Published", y="Count", color="Sentiment",
                            title="Sentiment Over Time", markers=True)
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("No valid timestamps available for trend plotting.")

    # ---- Word Cloud ----
    st.subheader("☁️ Word Cloud")
    if split_clouds:
        col1, col2 = st.columns(2)
        pos_text = text_join(df.loc[df["Sentiment"] == "Positive", "Text"])
        neg_text = text_join(df.loc[df["Sentiment"] == "Negative", "Text"])
        with col1:
            st.caption("Positive")
            if pos_text.strip():
                st.pyplot(make_wordcloud(pos_text, extra_stopwords={topic.lower()}))
            else:
                st.info("Not enough positive articles to build a cloud.")
        with col2:
            st.caption("Negative")
            if neg_text.strip():
                st.pyplot(make_wordcloud(neg_text, extra_stopwords={topic.lower()}))
            else:
                st.info("Not enough negative articles to build a cloud.")
    else:
        all_text = text_join(df["Text"])
        st.pyplot(make_wordcloud(all_text, extra_stopwords={topic.lower()}))

    # ---- Download CSV ----
    st.subheader("📂 Download Data")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download results as CSV",
        data=csv,
        file_name=f"{topic}_sentiment_news.csv",
        mime="text/csv",
    )

    # ---- Data Table ----
    st.subheader("🧾 News Articles")
    st.dataframe(df[["Published", "Title", "Sentiment", "Link"]], use_container_width=True)

    # ---- Footer / Disclaimer ----
    st.markdown("---")
    st.caption("© 2025 Astittva — Data Source: Google News RSS | For educational use only")

else:
    st.info("👆 Enter a topic, choose article count, and click **Fetch & Analyze News**. "
            "Use the checkbox to split the word cloud by sentiment.")
