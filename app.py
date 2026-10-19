import streamlit as st
import requests
import yfinance as yf
from datetime import datetime, timedelta
from transformers import pipeline

st.set_page_config(page_title="News + TA Bot", page_icon="📈", layout="wide")
st.title("📈 News + Technical Analysis Bot")
st.caption("Short-term (15m–2h) • Flexible confidence • Stronger news")

# ====================== SECRETS ======================
try:
    TELEGRAM_BOT_TOKEN = st.secrets["TELEGRAM_BOT_TOKEN"]
    TELEGRAM_CHAT_ID = st.secrets["TELEGRAM_CHAT_ID"]
    NEWS_API_KEY = st.secrets["NEWS_API_KEY"]
    st.sidebar.success("Keys loaded")
except Exception as e:
    st.error(f"Secrets error: {e}")
    st.stop()

send_telegram_alerts = st.sidebar.checkbox("Send Telegram Alerts", value=True)

if st.sidebar.button("📨 Test Telegram"):
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": "✅ Test successful"},
            timeout=10
        )
        st.sidebar.success("Telegram OK" if r.status_code == 200 else r.text)
    except Exception as e:
        st.sidebar.error(str(e))

# More flexible confidence
MIN_CONFIDENCE = 0.58

FULL_WATCHLIST = {
    "BTC-USD": {"name": "Bitcoin", "keywords": ["Bitcoin", "BTC", "Bitcoin price"], "must_contain": ["bitcoin", "btc"], "open_weekend": True},
    "GC=F": {"name": "Gold", "keywords": ["gold", "gold price", "XAUUSD", "bullion"], "must_contain": ["gold", "xau", "bullion"], "open_weekend": False},
    "SI=F": {"name": "Silver", "keywords": ["silver", "silver price", "XAGUSD"], "must_contain": ["silver", "xag"], "open_weekend": False},
    "CL=F": {"name": "Crude Oil", "keywords": ["oil", "crude oil", "WTI", "Brent"], "must_contain": ["oil", "wti", "brent", "crude"], "open_weekend": False},
    "EURUSD=X": {"name": "EUR/USD", "keywords": ["EURUSD", "EUR/USD", "euro"], "must_contain": ["eur", "euro"], "open_weekend": False},
    "GBPUSD=X": {"name": "GBP/USD", "keywords": ["GBPUSD", "GBP/USD", "pound"], "must_contain": ["gbp", "pound", "sterling"], "open_weekend": False},
    "AUDUSD=X": {"name": "AUD/USD", "keywords": ["AUDUSD", "AUD/USD", "aussie"], "must_contain": ["aud", "aussie", "australia"], "open_weekend": False},
    "USDJPY=X": {"name": "USD/JPY", "keywords": ["USDJPY", "USD/JPY", "yen"], "must_contain": ["jpy", "yen", "japan"], "open_weekend": False},
}

def is_weekend_now():
    return datetime.utcnow().weekday() >= 5

def get_active_watchlist():
    if is_weekend_now():