import streamlit as st
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from transformers import pipeline
import traceback

st.set_page_config(page_title="News + TA Bot", page_icon="📈", layout="wide")
st.title("📈 News + Technical Analysis Bot")
st.caption("Improved data loading + clear errors")

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

if st.sidebar.button("📨 Test Telegram Now"):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        r = requests.post(url, json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": "✅ Telegram test from improved bot"
        }, timeout=10)
        if r.status_code == 200:
            st.sidebar.success("Telegram message sent successfully!")
        else:
            st.sidebar.error(f"Telegram error: {r.text}")
    except Exception as e:
        st.sidebar.error(f"Failed: {e}")

WATCHLIST = {
    "GC=F": {"name": "Gold", "keywords": ["gold", "XAUUSD"]},
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

try:
    sentiment_model = load_model()
    st.sidebar.success("FinBERT ready")
except Exception as e:
    st.sidebar.error(f"FinBERT error: {e}")
    sentiment_model = None

def send_telegram(msg):
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"},
            timeout=10
        )
        return r.status_code == 200, r.text
    except Exception as e:
        return False, str(e)

def get_technical(symbol):
    """Try multiple methods to get price data"""
    errors = []
    
    # Method 1: 1h data
    try:
        df = yf.download(symbol, period="5d", interval="1h", progress=False, threads=False)
        if not df.empty and len(df) >= 20:
            df = df.dropna()
            close = df["Close"]
            price = float(close.iloc[-1])
            sma20 = float(close.rolling(20).mean().iloc[-1])
            
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = float(100 - (100 / (1 + rs.iloc[-1])))
            
            if price > sma20 and rsi >= 50:
                direction = "bullish"
            elif price < sma20 and rsi <= 50:
                direction = "bearish"
            else:
                direction = "neutral"
                
            reason = f"Price {price:.2f} | SMA20 {sma20:.2f} | RSI {rsi:.1f}"
            return {"price": price, "direction": direction, "reason": reason}
    except Exception as e:
        errors.append(f"1h: {str(e)[:60]}")

    # Method 2: Daily data fallback
    try:
        df = yf.download(symbol, period="1mo", interval="1d", progress=False, threads=False)
        if not df.empty and len(df) >= 10:
            df = df.dropna()
            close = df["Close"]
            price = float(close.iloc[-1])
            sma20 = float(close.rolling(min(20, len(close))).mean().iloc[-1])
            
            if price > sma20:
                direction = "bullish"
            else:
                direction = "bearish"
                
            reason = f"Daily Price {price:.2f} | SMA {sma20:.2f}"
            return {"price": price, "direction": direction, "reason": reason}
    except Exception as e:
        errors.append(f"1d: {str(e)[:60]}")

    # Method 3: Ticker info fallback
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.fast_info
        price = float(info["lastPrice"])
        return {
            "price": price,
            "direction": "neutral",
            "reason": f"Last price only: {price:.2f} (limited data)"
        }
    except Exception as e:
        errors.append(f"info: {str(e)[:60]}")

    return {"error": " | ".join(errors)}

def get_news(keywords):
    if not sentiment_model:
        return None
    query = " OR ".join(keywords)
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 6,
        "apiKey": NEWS_API_KEY,
        "from": (datetime.utcnow() - timedelta(hours=24)).isoformat()
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
            if label in ["positive", "negative"] and score >= 0.55:
                direction = "bullish" if label == "positive" else "bearish"
                if best is None or score > best["confidence"]:
                    best = {"direction": direction, "confidence": score, "headline": title}
        except:
            continue
    return best

# Session state
if "logs" not in st.session_state:
    st.session_state.logs = []
if "signals" not in st.session_state:
    st.session_state.signals = []

if st.button("🔄 Run Full Analysis", use_container_width=True):
    st.session_state.logs = []
    st.session_state.signals = []
    
    with st.spinner("Loading data and analyzing..."):
        for symbol, info in WATCHLIST.items():
            name = info["name"]
            st.session_state.logs.append(f"\n——— {name} ({symbol}) ———")

            # Technical
            tech = get_technical(symbol)
            
            if "error" in tech:
                st.session_state.logs.append(f"Technical ERROR: {tech['error']}")
                continue
                
            st.session_state.logs.append(f"Technical: {tech['reason']}")
            st.session_state.logs.append(f"TA Direction: {tech['direction'].upper()}")

            # News
            news = get_news(info["keywords"])
            if news:
                st.session_state.logs.append(f"News: {news['direction'].upper()} ({news['confidence']:.0%})")
                st.session_state.logs.append(f"Headline: {news['headline'][:70]}...")
                final_direction = news["direction"]
                source = "News + Technical"
                headline = news["headline"]
            else:
                st.session_state.logs.append("News: No strong headline found")
                final_direction = tech["direction"]
                source = "Technical only"
                headline = "No clear news"

            if final_direction == "neutral":
                st.session_state.logs.append("Result: Neutral – no signal")
                continue

            # Create signal
            price_text = f"${tech['price']:,.2f}"
            signal = {
                "name": name,
                "direction": final_direction,
                "price": price_text,
                "ta": tech["reason"],
                "headline": headline,
                "source": source,
                "time": datetime.utcnow().strftime("%H:%M UTC")
            }
            st.session_state.signals.append(signal)
            st.session_state.logs.append(f"SIGNAL: {final_direction.upper()} @ {price_text}")

            # Telegram
            if send_telegram_alerts:
                msg = (
                    f"{'🟢 BULLISH' if final_direction == 'bullish' else '🔴 BEARISH'} – {name}\n\n"
                    f"Price: {price_text}\n"
                    f"Technical: {tech['reason']}\n"
                    f"Source: {source}\n"
                    f"Time: {signal['time']}\n\n"
                    f"{headline}"
                )
                ok, resp = send_telegram(msg)
                if ok:
                    st.session_state.logs.append("Telegram: Sent successfully")
                else:
                    st.session_state.logs.append(f"Telegram FAILED: {resp[:100]}")

    st.success("Analysis finished – check log below")

# ====================== DISPLAY ======================
st.subheader("📊 Clear Market Directions")
if st.session_state.signals:
    for s in st.session_state.signals:
        color = "green" if s["direction"] == "bullish" else "red"
        st.markdown(
            f":{color}[**{s['direction'].upper()} – {s['name']}**]\n\n"
            f"**Price:** {s['price']}\n"
            f"**Technical Analysis:** {s['ta']}\n"
            f"**Source:** {s['source']}\n"
            f"**Time:** {s['time']}\n"
            f"{s['headline']}"
        )
else:
    st.info("No clear Bullish or Bearish signals yet.")

st.subheader("🔍 Detailed Log (Read this carefully)")
if st.session_state.logs:
    for line in st.session_state.logs:
        st.text(line)
else:
    st.write("Click **Run Full Analysis**")