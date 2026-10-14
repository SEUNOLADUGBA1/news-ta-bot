import streamlit as st
import requests
import yfinance as yf
from datetime import datetime, timedelta
from transformers import pipeline

st.set_page_config(page_title="News + TA Bot", page_icon="📈", layout="wide")
st.title("📈 News + Technical Analysis Bot")
st.caption("Works even with limited data")

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

WATCHLIST = {
    "GC=F": {"name": "Gold", "keywords": ["gold", "XAUUSD", "gold price"]},
    "SI=F": {"name": "Silver", "keywords": ["silver", "XAG"]},
    "CL=F": {"name": "Crude Oil", "keywords": ["oil", "crude", "WTI"]},
    "BTC-USD": {"name": "Bitcoin", "keywords": ["Bitcoin", "BTC"]},
    "EURUSD=X": {"name": "EUR/USD", "keywords": ["EURUSD", "EUR/USD"]},
    "GBPUSD=X": {"name": "GBP/USD", "keywords": ["GBPUSD", "GBP/USD"]},
    "AUDUSD=X": {"name": "AUD/USD", "keywords": ["AUDUSD", "AUD/USD"]},
    "USDJPY=X": {"name": "USD/JPY", "keywords": ["USDJPY", "yen"]},
}

@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert", truncation=True)

sentiment_model = load_model()

def send_telegram(msg):
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"},
            timeout=10
        )
        return r.status_code == 200
    except:
        return False

def get_price_and_simple_ta(symbol):
    """Get price + simple direction even with limited data"""
    try:
        # Try full data first
        df = yf.download(symbol, period="5d", interval="1h", progress=False)
        if not df.empty and len(df) >= 20:
            df = df.dropna()
            close = df["Close"]
            price = float(close.iloc[-1])
            sma20 = float(close.rolling(20).mean().iloc[-1])
            change = ((price - float(close.iloc[0])) / float(close.iloc[0])) * 100

            if price > sma20 and change > 0.3:
                direction = "bullish"
            elif price < sma20 and change < -0.3:
                direction = "bearish"
            else:
                direction = "neutral"

            reason = f"Price {price:.2f} | SMA20 {sma20:.2f} | Change {change:.2f}%"
            return price, direction, reason
    except:
        pass

    # Fallback – last price only
    try:
        ticker = yf.Ticker(symbol)
        price = float(ticker.fast_info["lastPrice"])
        # Very simple fallback direction using previous close if available
        try:
            prev = float(ticker.fast_info["previousClose"])
            change = ((price - prev) / prev) * 100
            if change > 0.4:
                direction = "bullish"
            elif change < -0.4:
                direction = "bearish"
            else:
                direction = "neutral"
            reason = f"Price {price:.2f} | vs Prev Close {change:.2f}%"
        except:
            direction = "neutral"
            reason = f"Price {price:.2f} (limited data)"
        return price, direction, reason
    except Exception as e:
        return None, "error", str(e)

def get_news(keywords):
    query = " OR ".join(keywords)
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 8,
        "apiKey": NEWS_API_KEY,
        "from": (datetime.utcnow() - timedelta(hours=36)).isoformat()
    }
    try:
        r = requests.get("https://newsapi.org/v2/everything", params=params, timeout=12)
        articles = r.json().get("articles", [])
    except:
        return None

    best = None
    for art in articles:
        title = art.get("title") or ""
        if len(title) < 12:
            continue
        try:
            result = sentiment_model(title[:400])[0]
            label = result["label"].lower()
            score = result["score"]
            if label in ["positive", "negative"] and score >= 0.50:  # more sensitive
                direction = "bullish" if label == "positive" else "bearish"
                if best is None or score > best["confidence"]:
                    best = {"direction": direction, "confidence": score, "headline": title}
        except:
            continue
    return best

if "logs" not in st.session_state:
    st.session_state.logs = []
if "signals" not in st.session_state:
    st.session_state.signals = []

if st.button("🔄 Run Analysis", use_container_width=True):
    st.session_state.logs = []
    st.session_state.signals = []

    with st.spinner("Analyzing..."):
        for symbol, info in WATCHLIST.items():
            name = info["name"]
            st.session_state.logs.append(f"——— {name} ———")

            price, ta_direction, ta_reason = get_price_and_simple_ta(symbol)

            if price is None:
                st.session_state.logs.append(f"ERROR: {ta_reason}")
                continue

            st.session_state.logs.append(f"Technical: {ta_reason}")
            st.session_state.logs.append(f"TA Direction: {ta_direction.upper()}")

            news = get_news(info["keywords"])

            if news:
                final_direction = news["direction"]
                source = "News + Technical"
                headline = news["headline"]
                conf = f"{news['confidence']:.0%}"
                st.session_state.logs.append(f"News: {final_direction.upper()} ({conf})")
                st.session_state.logs.append(f"Headline: {headline[:65]}...")
            else:
                final_direction = ta_direction
                source = "Technical only"
                headline = "No strong news found"
                st.session_state.logs.append("News: None")

            if final_direction in ["bullish", "bearish"]:
                price_text = f"${price:,.2f}"
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
                st.session_state.logs.append(f"✅ SIGNAL: {final_direction.upper()}")

                if send_telegram_alerts:
                    emoji = "🟢" if final_direction == "bullish" else "🔴"
                    msg = (
                        f"{emoji} <b>{final_direction.upper()}</b> – {name}\n\n"
                        f"Price: {price_text}\n"
                        f"Technical: {ta_reason}\n"
                        f"Source: {source}\n"
                        f"Time: {signal['time']}\n\n"
                        f"{headline}"
                    )
                    ok = send_telegram(msg)
                    st.session_state.logs.append("Telegram: Sent" if ok else "Telegram: Failed")
            else:
                st.session_state.logs.append("Result: Neutral – no signal")

    st.success("Analysis complete")

# Display
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
    st.info("No clear Bullish or Bearish signals at the moment.")

st.subheader("🔍 Log")
for line in st.session_state.logs:
    st.text(line)