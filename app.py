import streamlit as st
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from transformers import pipeline

st.set_page_config(page_title="News + TA Trading Bot", page_icon="📈", layout="wide")

st.title("📈 News + Technical Analysis Bot")
st.caption("Now includes Entry & Exit prices")

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
            st.sidebar.success("Telegram test sent successfully!")
        else:
            st.sidebar.error(f"Error: {r.text}")
    except Exception as e:
        st.sidebar.error(f"Failed: {e}")

CONFIDENCE_THRESHOLD = 0.30

WATCHLIST = {
    "BTC-USD": {"name": "Bitcoin", "keywords": ["Bitcoin", "BTC"]},
    "ETH-USD": {"name": "Ethereum", "keywords": ["Ethereum", "ETH"]},
    "SOL-USD": {"name": "Solana", "keywords": ["Solana", "SOL"]},
    "AAPL": {"name": "Apple", "keywords": ["Apple", "AAPL"]},
    "TSLA": {"name": "Tesla", "keywords": ["Tesla", "TSLA"]},
    "NVDA": {"name": "Nvidia", "keywords": ["Nvidia", "NVDA"]},
    "EURUSD=X": {"name": "EUR/USD", "keywords": ["EURUSD", "euro", "EUR/USD"]},
    "GBPUSD=X": {"name": "GBP/USD", "keywords": ["GBPUSD", "pound", "GBP/USD"]},
    "USDJPY=X": {"name": "USD/JPY", "keywords": ["USDJPY", "yen", "USD/JPY"]},
    "GC=F": {"name": "Gold", "keywords": ["gold", "XAU"]},
    "SI=F": {"name": "Silver", "keywords": ["silver", "XAG"]},
    "CL=F": {"name": "Crude Oil", "keywords": ["oil", "crude", "WTI"]},
    "NG=F": {"name": "Natural Gas", "keywords": ["natural gas", "gas"]},
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
        df = yf.download(symbol, period="1d", interval="1m", progress=False)
        if not df.empty:
            return float(df["Close"].iloc[-1])
    except:
        pass
    return None

def get_news_sentiment(keywords):
    query = " OR ".join(keywords)
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 12,
        "apiKey": NEWS_API_KEY,
        "from": (datetime.utcnow() - timedelta(hours=24)).isoformat()
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
        if label in ["positive", "negative"] and score >= CONFIDENCE_THRESHOLD:
            direction = "bullish" if label == "positive" else "bearish"
            if best is None or score > best["confidence"]:
                best = {"direction": direction, "confidence": score, "headline": title}
    return best

def get_technical_signal(symbol, direction):
    try:
        df = yf.download(symbol, period="5d", interval="1h", progress=False)
        if df.empty or len(df) < 20:
            return True, "Limited data"
        df["SMA20"] = df["Close"].rolling(20).mean()
        latest = df.iloc[-1]
        price = float(latest["Close"])
        sma20 = float(latest["SMA20"])
        if direction == "bullish":
            confirmed = price > sma20 * 0.99
            reason = f"Price {price:.4f} vs SMA20"
        else:
            confirmed = price < sma20 * 1.01
            reason = f"Price {price:.4f} vs SMA20"
        return confirmed, reason
    except:
        return True, "Technicals passed"

if "positions" not in st.session_state:
    st.session_state.positions = {}
if "signals" not in st.session_state:
    st.session_state.signals = []
if "logs" not in st.session_state:
    st.session_state.logs = []

if st.button("🔄 Run Analysis Now", use_container_width=True):
    st.session_state.logs = []
    with st.spinner("Scanning news + technicals..."):
        for symbol, info in WATCHLIST.items():
            name = info["name"]
            news = get_news_sentiment(info["keywords"])
            
            if not news:
                st.session_state.logs.append(f"{name}: No relevant news found")
                continue

            direction = news["direction"]
            conf = news["confidence"]
            headline = news["headline"]
            current_price = get_current_price(symbol)
            price_text = f"${current_price:,.4f}" if current_price else "N/A"

            st.session_state.logs.append(f"{name}: {direction.upper()} ({conf:.0%}) → {headline[:60]}...")

            # EXIT
            if symbol in st.session_state.positions:
                open_pos = st.session_state.positions[symbol]
                if (open_pos["side"] == "long" and direction == "bearish") or (open_pos["side"] == "short" and direction == "bullish"):
                    signal = {
                        "type": "EXIT",
                        "symbol": name,
                        "side": open_pos["side"],
                        "headline": headline,
                        "confidence": conf,
                        "price": price_text,
                        "time": datetime.utcnow().strftime("%Y-%m-%d %H:%M")
                    }
                    st.session_state.signals.insert(0, signal)
                    if send_telegram_alerts:
                        msg = (f"🔴 <b>EXIT {open_pos['side'].upper()}</b> – {name}\n"
                               f"Price: {price_text}\n"
                               f"News: {headline}\n"
                               f"Confidence: {conf:.0%}")
                        send_telegram(msg)
                    del st.session_state.positions[symbol]
                    st.session_state.logs.append(f"→ EXIT sent for {name} at {price_text}")
                    continue

            # ENTRY
            tech_ok, tech_reason = get_technical_signal(symbol, direction)
            if tech_ok and symbol not in st.session_state.positions:
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
                           f"Confidence: {conf:.0%}\n"
                           f"{tech_reason}")
                    ok = send_telegram(msg)
                    st.session_state.logs.append(f"→ ENTRY sent for {name} at {price_text}" + (" (OK)" if ok else " (failed)"))
            else:
                st.session_state.logs.append(f"→ Technicals rejected for {name}")

    st.success("Analysis complete!")

st.subheader("📊 Current Open Positions")
if st.session_state.positions:
    for sym, pos in st.session_state.positions.items():
        st.info(f"**{pos['name']}** | {pos['side'].upper()} | Entry: {pos.get('entry_price', 'N/A')} | {pos['time']}\n\n{pos['headline']}")
else:
    st.write("No open positions")

st.subheader("📜 Recent Signals")
if st.session_state.signals:
    for sig in st.session_state.signals[:12]:
        color = "green" if sig["type"] == "ENTRY" else "red"
        st.markdown(f":{color}[**{sig['type']} {sig['side'].upper()} – {sig['symbol']}**]  \nPrice: {sig.get('price', 'N/A')}  \n{sig['headline']}  \nConfidence: {sig['confidence']:.0%} | {sig['time']}")
else:
    st.write("No signals yet")

st.subheader("🔍 Analysis Log")
if st.session_state.logs:
    for log in st.session_state.logs:
        st.text(log)
else:
    st.write("Click **Run Analysis Now**")