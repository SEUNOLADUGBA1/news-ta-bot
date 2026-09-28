import streamlit as st
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from transformers import pipeline

st.set_page_config(page_title="News + TA Trading Bot", page_icon="📈", layout="wide")

st.title("📈 News + Technical Analysis Bot")
st.caption("Debug version – shows real headlines from NewsAPI")

# ====================== READ KEYS FROM SECRETS ======================
try:
    TELEGRAM_BOT_TOKEN = st.secrets["TELEGRAM_BOT_TOKEN"]
    TELEGRAM_CHAT_ID = st.secrets["TELEGRAM_CHAT_ID"]
    NEWS_API_KEY = st.secrets["NEWS_API_KEY"]
    st.sidebar.success("Keys loaded from Secrets")
except Exception as e:
    st.error("Could not load Secrets")
    st.stop()
# ===================================================================

send_telegram_alerts = st.sidebar.checkbox("Send Telegram Alerts", value=True)

if st.sidebar.button("📨 Test Telegram Connection"):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": "✅ Test message from your News + TA Bot!"}
    try:
        r = requests.post(url, json=payload, timeout=10)
        if r.status_code == 200:
            st.sidebar.success("Telegram test sent!")
        else:
            st.sidebar.error(f"Error: {r.text}")
    except Exception as e:
        st.sidebar.error(str(e))

CONFIDENCE_THRESHOLD = 0.28

WATCHLIST = {
    "BTC-USD": {"name": "Bitcoin", "keywords": ["Bitcoin", "BTC", "crypto"]},
    "ETH-USD": {"name": "Ethereum", "keywords": ["Ethereum", "ETH"]},
    "SOL-USD": {"name": "Solana", "keywords": ["Solana", "SOL"]},
    "AAPL": {"name": "Apple", "keywords": ["Apple", "AAPL", "iPhone"]},
    "TSLA": {"name": "Tesla", "keywords": ["Tesla", "TSLA", "Elon Musk"]},
    "NVDA": {"name": "Nvidia", "keywords": ["Nvidia", "NVDA", "AI chip"]},
    "EURUSD=X": {"name": "EUR/USD", "keywords": ["EURUSD", "euro", "EUR/USD", "euro dollar"]},
    "GBPUSD=X": {"name": "GBP/USD", "keywords": ["GBPUSD", "pound", "sterling"]},
    "USDJPY=X": {"name": "USD/JPY", "keywords": ["USDJPY", "yen", "dollar yen"]},
    "GC=F": {"name": "Gold", "keywords": ["gold", "XAUUSD", "precious metal"]},
    "SI=F": {"name": "Silver", "keywords": ["silver", "XAG"]},
    "CL=F": {"name": "Crude Oil", "keywords": ["oil", "crude oil", "WTI", "Brent"]},
    "NG=F": {"name": "Natural Gas", "keywords": ["natural gas", "gas prices"]},
}

@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert", truncation=True)

sentiment_pipeline = load_sentiment_model()

def send_telegram(message: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        r = requests.post(url, json=payload, timeout=10)
        return r.status_code == 200
    except:
        return False

def get_current_price(symbol):
    try:
        df = yf.download(symbol, period="1d", interval="5m", progress=False)
        if not df.empty:
            return float(df["Close"].iloc[-1])
    except:
        pass
    return None

def get_news(keywords):
    query = " OR ".join(keywords)
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 8,
        "apiKey": NEWS_API_KEY,
        "from": (datetime.utcnow() - timedelta(hours=48)).isoformat()
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        data = r.json()
        articles = data.get("articles", [])
        return articles
    except Exception as e:
        st.warning(f"NewsAPI error: {e}")
        return []

if "positions" not in st.session_state:
    st.session_state.positions = {}
if "signals" not in st.session_state:
    st.session_state.signals = []
if "logs" not in st.session_state:
    st.session_state.logs = []

if st.button("🔄 Run Analysis Now", use_container_width=True):
    st.session_state.logs = []
    with st.spinner("Fetching news from NewsAPI..."):
        for symbol, info in WATCHLIST.items():
            name = info["name"]
            articles = get_news(info["keywords"])

            if not articles:
                st.session_state.logs.append(f"{name}: No articles returned by NewsAPI")
                continue

            # Show the top headlines so we can see what is coming
            st.session_state.logs.append(f"——— {name} ———")
            best = None

            for art in articles[:5]:
                title = art.get("title") or ""
                if not title:
                    continue
                st.session_state.logs.append(f"  • {title[:90]}")

                result = sentiment_pipeline(title[:512])[0]
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

            if not best:
                st.session_state.logs.append(f"{name}: Headlines found but none passed sentiment threshold")
                continue

            direction = best["direction"]
            conf = best["confidence"]
            headline = best["headline"]
            price = get_current_price(symbol)
            price_text = f"${price:,.4f}" if price else "N/A"

            st.session_state.logs.append(f"→ Selected: {direction.upper()} ({conf:.0%})")

            # ENTRY (simplified for debugging)
            if symbol not in st.session_state.positions:
                side = "long" if direction == "bullish" else "short"
                st.session_state.positions[symbol] = {
                    "side": side,
                    "name": name,
                    "headline": headline,
                    "confidence": conf,
                    "entry_price": price_text,
                    "time": datetime.utcnow().strftime("%Y-%m-%d %H:%M")
                }
                signal = {
                    "type": "ENTRY",
                    "symbol": name,
                    "side": side,
                    "headline": headline,
                    "confidence": conf,
                    "price": price_text,
                    "time": datetime.utcnow().strftime("%Y-%m-%d %H:%M")
                }
                st.session_state.signals.insert(0, signal)

                if send_telegram_alerts:
                    msg = (f"🟢 <b>ENTRY {side.upper()}</b> – {name}\n"
                           f"Price: {price_text}\n"
                           f"News: {headline}\n"
                           f"Confidence: {conf:.0%}")
                    ok = send_telegram(msg)
                    st.session_state.logs.append(f"→ ENTRY sent at {price_text}" + (" (Telegram OK)" if ok else " (Telegram failed)"))

    st.success("Analysis finished – check the log below")

st.subheader("📊 Current Open Positions")
if st.session_state.positions:
    for sym, pos in st.session_state.positions.items():
        st.info(f"**{pos['name']}** | {pos['side'].upper()} | Entry: {pos.get('entry_price', 'N/A')}\n\n{pos['headline']}")
else:
    st.write("No open positions")

st.subheader("📜 Recent Signals")
if st.session_state.signals:
    for sig in st.session_state.signals[:10]:
        color = "green" if sig["type"] == "ENTRY" else "red"
        st.markdown(f":{color}[**{sig['type']} {sig['side'].upper()} – {sig['symbol']}**]  \nPrice: {sig.get('price', 'N/A')}  \n{sig['headline']}")
else:
    st.write("No signals yet")

st.subheader("🔍 Detailed Analysis Log")
if st.session_state.logs:
    for log in st.session_state.logs:
        st.text(log)
else:
    st.write("Click **Run Analysis Now**")