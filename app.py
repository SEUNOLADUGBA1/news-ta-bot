import streamlit as st
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from transformers import pipeline

st.set_page_config(page_title="News + TA Bot – Best Version", page_icon="📈", layout="wide")

st.title("📈 News + Technical Analysis Bot")
st.caption("Best Version – Very Clear News + Market Structure Levels")

# ====================== SECRETS ======================
try:
    TELEGRAM_BOT_TOKEN = st.secrets["TELEGRAM_BOT_TOKEN"]
    TELEGRAM_CHAT_ID = st.secrets["TELEGRAM_CHAT_ID"]
    NEWS_API_KEY = st.secrets["NEWS_API_KEY"]
    st.sidebar.success("Keys loaded successfully")
except:
    st.error("Secrets not found. Please check Streamlit Secrets.")
    st.stop()

send_telegram_alerts = st.sidebar.checkbox("Send Telegram Alerts", value=True)

if st.sidebar.button("📨 Test Telegram Connection"):
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": "✅ Bot is working correctly"},
            timeout=10
        )
        if r.status_code == 200:
            st.sidebar.success("Telegram test successful")
        else:
            st.sidebar.error(r.text)
    except Exception as e:
        st.sidebar.error(str(e))

# Very clear news only
CONFIDENCE_THRESHOLD = 0.72

WATCHLIST = {
    "BTC-USD": {"name": "Bitcoin", "keywords": ["Bitcoin", "BTC price"]},
    "ETH-USD": {"name": "Ethereum", "keywords": ["Ethereum", "ETH price"]},
    "SOL-USD": {"name": "Solana", "keywords": ["Solana", "SOL price"]},
    "AAPL": {"name": "Apple", "keywords": ["Apple stock", "AAPL"]},
    "TSLA": {"name": "Tesla", "keywords": ["Tesla stock", "TSLA"]},
    "NVDA": {"name": "Nvidia", "keywords": ["Nvidia stock", "NVDA"]},
    "EURUSD=X": {"name": "EUR/USD", "keywords": ["EURUSD", "EUR/USD"]},
    "GBPUSD=X": {"name": "GBP/USD", "keywords": ["GBPUSD", "GBP/USD"]},
    "USDJPY=X": {"name": "USD/JPY", "keywords": ["USDJPY", "USD/JPY"]},
    "GC=F": {"name": "Gold", "keywords": ["gold price", "XAUUSD", "gold"]},
    "SI=F": {"name": "Silver", "keywords": ["silver price", "XAG"]},
    "CL=F": {"name": "Crude Oil", "keywords": ["crude oil", "WTI", "oil price"]},
    "NG=F": {"name": "Natural Gas", "keywords": ["natural gas", "gas price"]},
}

@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert", truncation=True)

sentiment_model = load_sentiment_model()

def send_telegram(message):
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"},
            timeout=10
        )
        return r.status_code == 200
    except:
        return False

def get_market_structure(symbol):
    try:
        df = yf.download(symbol, period="7d", interval="1h", progress=False)
        if df.empty or len(df) < 30:
            return None

        df = df.dropna()
        current_price = float(df["Close"].iloc[-1])

        # ATR
        high = df["High"]
        low = df["Low"]
        close = df["Close"]
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        atr = float(tr.rolling(14).mean().iloc[-1])

        recent_high = float(high.tail(30).max())
        recent_low = float(low.tail(30).min())

        return {
            "price": current_price,
            "atr": atr,
            "recent_high": recent_high,
            "recent_low": recent_low
        }
    except:
        return None

def calculate_levels(structure, side):
    if not structure:
        return None, None

    price = structure["price"]
    atr = structure["atr"]
    high = structure["recent_high"]
    low = structure["recent_low"]

    if side == "long":
        stop_loss = min(low, price - 1.8 * atr)
        take_profit = price + (price - stop_loss) * 1.8   # good risk-reward
    else:
        stop_loss = max(high, price + 1.8 * atr)
        take_profit = price - (stop_loss - price) * 1.8

    return stop_loss, take_profit

def get_very_clear_news(keywords):
    query = " OR ".join(keywords)
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 12,
        "apiKey": NEWS_API_KEY,
        "from": (datetime.utcnow() - timedelta(hours=20)).isoformat()
    }
    try:
        response = requests.get(url, params=params, timeout=12)
        articles = response.json().get("articles", [])
    except:
        return None

    best = None
    for article in articles:
        title = article.get("title") or ""
        if len(title) < 25:
            continue

        result = sentiment_model(title[:500])[0]
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
if "positions" not in st.session_state:
    st.session_state.positions = {}
if "signals" not in st.session_state:
    st.session_state.signals = []
if "logs" not in st.session_state:
    st.session_state.logs = []

if st.button("🔄 Run Best Analysis", use_container_width=True):
    st.session_state.logs = []
    with st.spinner("Analyzing clear news + market structure..."):
        for symbol, info in WATCHLIST.items():
            name = info["name"]
            news = get_very_clear_news(info["keywords"])

            if not news:
                st.session_state.logs.append(f"{name}: No very clear news")
                continue

            direction = news["direction"]
            confidence = news["confidence"]
            headline = news["headline"]
            structure = get_market_structure(symbol)

            if not structure:
                st.session_state.logs.append(f"{name}: Could not get price data")
                continue

            price = structure["price"]
            price_text = f"${price:,.2f}"
            entry_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

            st.session_state.logs.append(f"{name}: CLEAR {direction.upper()} ({confidence:.0%}) @ {price_text}")
            st.session_state.logs.append(f"   {headline[:85]}...")

            # EXIT logic
            if symbol in st.session_state.positions:
                pos = st.session_state.positions[symbol]
                should_exit = (pos["side"] == "long" and direction == "bearish") or \
                              (pos["side"] == "short" and direction == "bullish")

                if should_exit:
                    signal = {
                        "type": "EXIT",
                        "symbol": name,
                        "side": pos["side"],
                        "price": price_text,
                        "headline": headline,
                        "time": entry_time,
                        "confidence": confidence
                    }
                    st.session_state.signals.insert(0, signal)

                    if send_telegram_alerts:
                        msg = (
                            f"🔴 <b>EXIT {pos['side'].upper()}</b> – {name}\n"
                            f"Exit Price: {price_text}\n"
                            f"Time: {entry_time}\n"
                            f"Reason: {headline}"
                        )
                        send_telegram(msg)

                    del st.session_state.positions[symbol]
                    st.session_state.logs.append(f"→ EXIT sent at {price_text}")
                    continue

            # ENTRY logic
            if symbol not in st.session_state.positions:
                side = "long" if direction == "bullish" else "short"
                stop_loss, take_profit = calculate_levels(structure, side)

                sl_text = f"${stop_loss:,.2f}" if stop_loss else "N/A"
                tp_text = f"${take_profit:,.2f}" if take_profit else "N/A"

                st.session_state.positions[symbol] = {
                    "side": side,
                    "name": name,
                    "entry_price": price_text,
                    "stop_loss": sl_text,
                    "take_profit": tp_text,
                    "headline": headline,
                    "confidence": confidence,
                    "time": entry_time
                }

                signal = {
                    "type": "ENTRY",
                    "symbol": name,
                    "side": side,
                    "price": price_text,
                    "stop_loss": sl_text,
                    "take_profit": tp_text,
                    "headline": headline,
                    "time": entry_time,
                    "confidence": confidence
                }
                st.session_state.signals.insert(0, signal)

                if send_telegram_alerts:
                    msg = (
                        f"🟢 <b>ENTRY {side.upper()}</b> – {name}\n\n"
                        f"Entry Price: {price_text}\n"
                        f"Stop Loss: {sl_text}\n"
                        f"Take Profit: {tp_text}\n"
                        f"Time: {entry_time}\n\n"
                        f"News: {headline}\n"
                        f"Confidence: {confidence:.0%}"
                    )
                    success = send_telegram(msg)
                    st.session_state.logs.append(
                        f"→ ENTRY sent | {price_text} | SL {sl_text} | TP {tp_text}" +
                        (" ✓" if success else " (Telegram failed)")
                    )

    st.success("Best analysis completed")

# ====================== DISPLAY ======================
st.subheader("📊 Open Positions")
if st.session_state.positions:
    for symbol, pos in st.session_state.positions.items():
        st.success(
            f"**{pos['name']}** — {pos['side'].upper()}\n\n"
            f"Entry: {pos['entry_price']}\n"
            f"Stop Loss: {pos['stop_loss']}\n"
            f"Take Profit: {pos['take_profit']}\n"
            f"Time: {pos['time']}\n\n"
            f"{pos['headline']}"
        )
else:
    st.info("No open positions")

st.subheader("📜 Recent Signals")
if st.session_state.signals:
    for sig in st.session_state.signals[:8]:
        color = "green" if sig["type"] == "ENTRY" else "red"
        extra = ""
        if sig["type"] == "ENTRY":
            extra = f"\nSL: {sig.get('stop_loss')} | TP: {sig.get('take_profit')}"
        st.markdown(
            f":{color}[**{sig['type']} {sig['side'].upper()} – {sig['symbol']}**]\n\n"
            f"Price: {sig['price']}{extra}\n"
            f"Time: {sig['time']}\n"
            f"{sig['headline']}"
        )
else:
    st.write("No signals yet")

st.subheader("🔍 Analysis Log")
if st.session_state.logs:
    for log in st.session_state.logs:
        st.text(log)
else:
    st.write("Click the button above to run analysis")