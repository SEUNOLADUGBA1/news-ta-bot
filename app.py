import streamlit as st
import time
import json
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from transformers import pipeline

st.set_page_config(page_title="News + TA Trading Bot", page_icon="📈", layout="wide")

st.title("📈 News + Technical Analysis Bot")
st.caption("News-driven signals confirmed by technical analysis | Crypto • Forex • Stocks • Energies")

st.sidebar.header("🔑 API Keys")
TELEGRAM_BOT_TOKEN = st.sidebar.text_input("Telegram Bot Token", type="password")
TELEGRAM_CHAT_ID = st.sidebar.text_input("Telegram Chat ID")
NEWS_API_KEY = st.sidebar.text_input("NewsAPI Key", type="password")
send_telegram_alerts = st.sidebar.checkbox("Send Telegram Alerts", value=True)

WATCHLIST = {
    "AAPL": {"class": "stock", "keywords": ["Apple", "AAPL"]},
    "TSLA": {"class": "stock", "keywords": ["Tesla", "TSLA"]},
    "BTC-USD": {"class": "crypto", "keywords": ["Bitcoin", "BTC"]},
    "ETH-USD": {"class": "crypto", "keywords": ["Ethereum", "ETH"]},
    "EURUSD=X": {"class": "forex", "keywords": ["EURUSD", "euro dollar"]},
    "GBPUSD=X": {"class": "forex", "keywords": ["GBPUSD", "pound dollar"]},
    "CL=F": {"class": "energy", "keywords": ["oil", "crude", "WTI"]},
    "GC=F": {"class": "energy", "keywords": ["gold"]},
}

CONFIDENCE_THRESHOLD = 0.75

@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert", truncation=True)

sentiment_pipeline = load_sentiment_model()

def send_telegram(message: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        requests.post(url, json=payload, timeout=10)
    except:
        pass

def get_news_sentiment(keywords):
    if not NEWS_API_KEY:
        return None
    query = " OR ".join(keywords)
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 6,
        "apiKey": NEWS_API_KEY,
        "from": (datetime.utcnow() - timedelta(hours=4)).isoformat()
    }
    try:
        r = requests.get(url, params=params, timeout=12)
        articles = r.json().get("articles", [])
    except:
        return None

    best = None
    for art in articles:
        title = art.get("title") or ""
        if not title:
            continue
        result = sentiment_pipeline(title[:512])[0]
        label = result["label"].lower()
        score = result["score"]
        if label == "positive" and score >= CONFIDENCE_THRESHOLD:
            if best is None or score > best["confidence"]:
                best = {"direction": "bullish", "confidence": score, "headline": title}
        elif label == "negative" and score >= CONFIDENCE_THRESHOLD:
            if best is None or score > best["confidence"]:
                best = {"direction": "bearish", "confidence": score, "headline": title}
    return best

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def get_technical_signal(symbol, direction):
    try:
        df = yf.download(symbol, period="5d", interval="1h", progress=False)
        if df.empty or len(df) < 25:
            return False, "Not enough data"
        df["SMA20"] = df["Close"].rolling(20).mean()
        df["RSI"] = compute_rsi(df["Close"])
        latest = df.iloc[-1]
        price = float(latest["Close"])
        sma20 = float(latest["SMA20"])
        rsi = float(latest["RSI"])
        if direction == "bullish":
            confirmed = price > sma20 and rsi < 70
            reason = f"Price {price:.4f} > SMA20 | RSI {rsi:.1f}"
        else:
            confirmed = price < sma20 and rsi > 30
            reason = f"Price {price:.4f} < SMA20 | RSI {rsi:.1f}"
        return confirmed, reason
    except Exception as e:
        return False, str(e)

if "positions" not in st.session_state:
    st.session_state.positions = {}
if "signals" not in st.session_state:
    st.session_state.signals = []

col1, col2 = st.columns([1, 3])
with col1:
    if st.button("🔄 Run Analysis Now", use_container_width=True):
        with st.spinner("Scanning news + technicals..."):
            for symbol, info in WATCHLIST.items():
                news = get_news_sentiment(info["keywords"])
                if not news:
                    continue
                direction = news["direction"]
                conf = news["confidence"]
                headline = news["headline"]

                if symbol in st.session_state.positions:
                    open_pos = st.session_state.positions[symbol]
                    if (open_pos["side"] == "long" and direction == "bearish") or (open_pos["side"] == "short" and direction == "bullish"):
                        signal = {"type": "EXIT", "symbol": symbol, "side": open_pos["side"], "headline": headline, "confidence": conf, "time": datetime.utcnow().strftime("%Y-%m-%d %H:%M")}
                        st.session_state.signals.insert(0, signal)
                        if send_telegram_alerts:
                            send_telegram(f"🔴 EXIT {open_pos['side'].upper()} – {symbol}\n{headline}")
                        del st.session_state.positions[symbol]
                        continue

                tech_ok, tech_reason = get_technical_signal(symbol, direction)
                if tech_ok and symbol not in st.session_state.positions:
                    side = "long" if direction == "bullish" else "short"
                    st.session_state.positions[symbol] = {"side": side, "headline": headline, "confidence": conf, "time": datetime.utcnow().strftime("%Y-%m-%d %H:%M")}
                    signal = {"type": "ENTRY", "symbol": symbol, "side": side, "headline": headline, "confidence": conf, "tech": tech_reason, "time": datetime.utcnow().strftime("%Y-%m-%d %H:%M")}
                    st.session_state.signals.insert(0, signal)
                    if send_telegram_alerts:
                        send_telegram(f"🟢 ENTRY {side.upper()} – {symbol}\n{headline}\n{tech_reason}")
            st.success("Analysis complete!")

st.subheader("📊 Current Open Positions")
if st.session_state.positions:
    for sym, pos in st.session_state.positions.items():
        st.info(f"**{sym}** | {pos['side'].upper()} | Conf: {pos['confidence']:.0%} | {pos['time']}\n\n{pos['headline']}")
else:
    st.write("No open positions")

st.subheader("📜 Recent Signals")
if st.session_state.signals:
    for sig in st.session_state.signals[:15]:
        color = "green" if sig["type"] == "ENTRY" else "red"
        st.markdown(f":{color}[**{sig['type']} {sig['side'].upper()} – {sig['symbol']}**]  \n{sig['headline']}  \nConfidence: {sig['confidence']:.0%} | {sig['time']}")
else:
    st.write("No signals yet. Click **Run Analysis Now**")

st.markdown("---")
st.caption("Bot only triggers on strong news + technical confirmation")
