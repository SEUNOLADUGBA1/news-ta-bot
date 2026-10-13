import streamlit as st
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from transformers import pipeline

st.set_page_config(page_title="News + TA Bot", page_icon="📈", layout="wide")
st.title("📈 News + Technical Analysis Bot")
st.caption("Shows clear market direction + Technical Analysis")

# ====================== SECRETS ======================
try:
    TELEGRAM_BOT_TOKEN = st.secrets["TELEGRAM_BOT_TOKEN"]
    TELEGRAM_CHAT_ID = st.secrets["TELEGRAM_CHAT_ID"]
    NEWS_API_KEY = st.secrets["NEWS_API_KEY"]
    st.sidebar.success("Keys loaded")
except Exception as e:
    st.error("Secrets missing. Please check Streamlit Secrets.")
    st.stop()

send_telegram_alerts = st.sidebar.checkbox("Send Telegram Alerts", value=True)

if st.sidebar.button("📨 Test Telegram"):
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": "✅ Bot is working"},
            timeout=10
        )
        st.sidebar.success("Telegram OK" if r.status_code == 200 else r.text)
    except Exception as e:
        st.sidebar.error(str(e))

CONFIDENCE_THRESHOLD = 0.55

WATCHLIST = {
    "GC=F": {"name": "Gold", "keywords": ["gold", "XAUUSD", "gold price"]},
    "SI=F": {"name": "Silver", "keywords": ["silver", "XAGUSD"]},
    "CL=F": {"name": "Crude Oil", "keywords": ["oil", "crude", "WTI"]},
    "BTC-USD": {"name": "Bitcoin", "keywords": ["Bitcoin", "BTC"]},
    "EURUSD=X": {"name": "EUR/USD", "keywords": ["EURUSD", "EUR/USD"]},
    "GBPUSD=X": {"name": "GBP/USD", "keywords": ["GBPUSD", "GBP/USD"]},
    "AUDUSD=X": {"name": "AUD/USD", "keywords": ["AUDUSD", "AUD/USD"]},
    "USDJPY=X": {"name": "USD/JPY", "keywords": ["USDJPY", "USD/JPY"]},
}

@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert", truncation=True)

sentiment_model = load_model()

def send_telegram(msg):
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"},
            timeout=10
        )
        return True
    except:
        return False

def get_technical(symbol):
    try:
        df = yf.download(symbol, period="7d", interval="1h", progress=False)
        if df.empty or len(df) < 30:
            return None

        df = df.dropna()
        close = df["Close"]
        high = df["High"]
        low = df["Low"]

        price = float(close.iloc[-1])
        sma20 = float(close.rolling(20).mean().iloc[-1])
        sma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else sma20

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = float(100 - (100 / (1 + rs.iloc[-1])))

        # Direction from technicals
        if price > sma20 and rsi > 50:
            direction = "bullish"
            reason = f"Price {price:.2f} > SMA20 | RSI {rsi:.1f} (Bullish)"
        elif price < sma20 and rsi < 50:
            direction = "bearish"
            reason = f"Price {price:.2f} < SMA20 | RSI {rsi:.1f} (Bearish)"
        else:
            direction = "neutral"
            reason = f"Price {price:.2f} | SMA20 {sma20:.2f} | RSI {rsi:.1f} (Neutral)"

        return {
            "price": price,
            "direction": direction,
            "reason": reason,
            "rsi": rsi,
            "sma20": sma20
        }
    except Exception as e:
        return None

def get_news_direction(keywords):
    query = " OR ".join(keywords)
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 8,
        "apiKey": NEWS_API_KEY,
        "from": (datetime.utcnow() - timedelta(hours=18)).isoformat()
    }
    try:
        r = requests.get("https://newsapi.org/v2/everything", params=params, timeout=12)
        articles = r.json().get("articles", [])
    except:
        return None

    best = None
    for art in articles:
        title = art.get("title") or ""
        if len(title) < 15:
            continue
        result = sentiment_model(title[:450])[0]
        label = result["label"].lower()
        score = result["score"]

        if label in ["positive", "negative"] and score >= CONFIDENCE_THRESHOLD:
            direction = "bullish" if label == "positive" else "bearish"
            if best is None or score > best["confidence"]:
                best = {
                    "direction": direction,
                    "confidence": score,
                    "headline": title
                }
    return best

# Session state
if "logs" not in st.session_state:
    st.session_state.logs = []
if "signals" not in st.session_state:
    st.session_state.signals = []

if st.button("🔄 Run Analysis", use_container_width=True):
    st.session_state.logs = []
    st.session_state.signals = []

    with st.spinner("Analyzing news + technicals..."):
        for symbol, info in WATCHLIST.items():
            name = info["name"]

            # Technical Analysis
            tech = get_technical(symbol)
            if not tech:
                st.session_state.logs.append(f"{name}: Could not load technical data")
                continue

            # News
            news = get_news_direction(info["keywords"])

            # Final decision
            if news:
                final_direction = news["direction"]
                source = "News + TA"
                headline = news["headline"]
                conf = news["confidence"]
            else:
                final_direction = tech["direction"]
                source = "Technical only"
                headline = "No strong news found"
                conf = 0.60

            price_text = f"${tech['price']:,.2f}"
            ta_reason = tech["reason"]

            # Log everything clearly
            st.session_state.logs.append(f"——— {name} ———")
            st.session_state.logs.append(f"Direction: {final_direction.upper()}")
            st.session_state.logs.append(f"Price: {price_text}")
            st.session_state.logs.append(f"Technical: {ta_reason}")
            st.session_state.logs.append(f"Source: {source}")
            if news:
                st.session_state.logs.append(f"News: {headline[:70]}...")

            # Only create signal if direction is clear (not neutral)
            if final_direction in ["bullish", "bearish"]:
                signal = {
                    "name": name,
                    "direction": final_direction,
                    "price": price_text,
                    "ta": ta_reason,
                    "headline": headline,
                    "source": source,
                    "time": datetime.utcnow().strftime("%H:%M UTC")
                }
                st.session_state.signals.append(signal)

                # Telegram
                if send_telegram_alerts:
                    msg = (
                        f"{'🟢' if final_direction == 'bullish' else '🔴'} "
                        f"<b>{final_direction.upper()}</b> – {name}\n\n"
                        f"Price: {price_text}\n"
                        f"Technical: {ta_reason}\n"
                        f"Source: {source}\n"
                        f"Time: {signal['time']}\n\n"
                        f"{headline}"
                    )
                    send_telegram(msg)

    st.success("Analysis complete")

# ====================== DISPLAY ======================
st.subheader("📊 Market Directions")
if st.session_state.signals:
    for s in st.session_state.signals:
        color = "green" if s["direction"] == "bullish" else "red"
        st.markdown(
            f":{color}[**{s['direction'].upper()} – {s['name']}**]\n\n"
            f"Price: {s['price']}\n"
            f"Technical: {s['ta']}\n"
            f"Source: {s['source']}\n"
            f"Time: {s['time']}\n"
            f"{s['headline']}"
        )
else:
    st.info("No clear directional signals yet. Click Run Analysis.")

st.subheader("🔍 Full Analysis Log")
if st.session_state.logs:
    for line in st.session_state.logs:
        st.text(line)
else:
    st.write("Click **Run Analysis** to see directions and technical analysis")