import streamlit as st
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from transformers import pipeline
import time

st.set_page_config(page_title="News + TA Bot Pro", page_icon="📈", layout="wide")
st.title("📈 News + Technical Analysis Bot – Pro")
st.caption("Auto-refresh every 1 minute | Smart Event Reminders")

# ====================== SECRETS ======================
try:
    TELEGRAM_BOT_TOKEN = st.secrets["TELEGRAM_BOT_TOKEN"]
    TELEGRAM_CHAT_ID = st.secrets["TELEGRAM_CHAT_ID"]
    NEWS_API_KEY = st.secrets["NEWS_API_KEY"]
    st.sidebar.success("Keys loaded")
except:
    st.error("Secrets missing – add them in Streamlit Settings")
    st.stop()

send_telegram_alerts = st.sidebar.checkbox("Send Telegram Alerts", value=True)
auto_refresh = st.sidebar.checkbox("Auto Refresh (1 min)", value=True)

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

CONFIDENCE_THRESHOLD = 0.68

WATCHLIST = {
    "BTC-USD": {"name": "Bitcoin", "keywords": ["Bitcoin", "BTC"]},
    "ETH-USD": {"name": "Ethereum", "keywords": ["Ethereum", "ETH"]},
    "SOL-USD": {"name": "Solana", "keywords": ["Solana", "SOL"]},
    "AAPL": {"name": "Apple", "keywords": ["Apple stock", "AAPL"]},
    "TSLA": {"name": "Tesla", "keywords": ["Tesla stock", "TSLA"]},
    "NVDA": {"name": "Nvidia", "keywords": ["Nvidia", "NVDA"]},
    "EURUSD=X": {"name": "EUR/USD", "keywords": ["EURUSD", "EUR/USD"]},
    "GBPUSD=X": {"name": "GBP/USD", "keywords": ["GBPUSD", "GBP/USD"]},
    "AUDUSD=X": {"name": "AUD/USD", "keywords": ["AUDUSD", "AUD/USD"]},
    "USDJPY=X": {"name": "USD/JPY", "keywords": ["USDJPY", "USD/JPY"]},
    "GC=F": {"name": "Gold (XAU/USD)", "keywords": ["gold price", "XAUUSD", "gold"]},
    "SI=F": {"name": "Silver (XAG/USD)", "keywords": ["silver price", "XAGUSD", "silver"]},
    "CL=F": {"name": "Crude Oil", "keywords": ["crude oil", "WTI", "oil price"]},
    "NG=F": {"name": "Natural Gas", "keywords": ["natural gas", "gas price"]},
}

# High Impact Events (from your calendar)
HIGH_IMPACT_EVENTS = [
    {"date": "2026-09-30", "time": "02:30", "event": "Australia CPI", "impact": "High"},
    {"date": "2026-10-02", "time": "12:30", "event": "US NFP", "impact": "Very High"},
    {"date": "2026-10-02", "time": "10:00", "event": "EU Flash CPI", "impact": "High"},
    {"date": "2026-10-14", "time": "13:30", "event": "US CPI", "impact": "Very High"},
    {"date": "2026-10-15", "time": "13:30", "event": "US PPI", "impact": "High"},
    {"date": "2026-10-28", "time": "18:00", "event": "FOMC Decision", "impact": "Very High"},
    {"date": "2026-10-29", "time": "13:15", "event": "ECB Decision", "impact": "Very High"},
    {"date": "2026-11-05", "time": "12:00", "event": "BoE Decision", "impact": "High"},
    {"date": "2026-11-06", "time": "12:30", "event": "US NFP", "impact": "Very High"},
]

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

def get_structure(symbol):
    try:
        df = yf.download(symbol, period="10d", interval="1h", progress=False)
        if df.empty or len(df) < 40:
            return None
        df = df.dropna()
        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        price = float(close.iloc[-1])
        sma20 = float(close.rolling(20).mean().iloc[-1])
        sma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else sma20
        tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
        atr = float(tr.rolling(14).mean().iloc[-1])
        recent_high = float(high.tail(36).max())
        recent_low = float(low.tail(36).min())
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = float(100 - (100 / (1 + rs.iloc[-1])))
        return {
            "price": price, "sma20": sma20, "sma50": sma50, "atr": atr,
            "recent_high": recent_high, "recent_low": recent_low, "rsi": rsi
        }
    except:
        return None

def calc_sl_tp(structure, side):
    if not structure:
        return None, None
    price = structure["price"]
    atr = structure["atr"]
    if side == "long":
        sl = min(structure["recent_low"], price - 1.7 * atr)
        tp = price + (price - sl) * 1.8
    else:
        sl = max(structure["recent_high"], price + 1.7 * atr)
        tp = price - (sl - price) * 1.8
    return sl, tp

def get_clear_news(keywords):
    query = " OR ".join(keywords)
    params = {
        "q": query, "language": "en", "sortBy": "publishedAt",
        "pageSize": 8, "apiKey": NEWS_API_KEY,
        "from": (datetime.utcnow() - timedelta(hours=16)).isoformat()
    }
    try:
        r = requests.get("https://newsapi.org/v2/everything", params=params, timeout=12)
        articles = r.json().get("articles", [])
    except:
        return None

    best = None
    for art in articles:
        title = art.get("title") or ""
        if len(title) < 22:
            continue
        res = sentiment_model(title[:500])[0]
        label = res["label"].lower()
        score = res["score"]
        if label in ["positive", "negative"] and score >= CONFIDENCE_THRESHOLD:
            direction = "bullish" if label == "positive" else "bearish"
            if best is None or score > best["confidence"]:
                best = {"direction": direction, "confidence": score, "headline": title, "source": "news"}
    return best

def get_strong_technical(structure):
    if not structure:
        return None
    price = structure["price"]
    sma20 = structure["sma20"]
    sma50 = structure["sma50"]
    rsi = structure["rsi"]

    if price > sma20 > sma50 and 52 < rsi < 72:
        return {"direction": "bullish", "confidence": 0.78, "headline": "Strong bullish technical structure", "source": "technical"}
    if price < sma20 < sma50 and 28 < rsi < 48:
        return {"direction": "bearish", "confidence": 0.78, "headline": "Strong bearish technical structure", "source": "technical"}
    # Extra gold sensitivity
    if price < structure["recent_low"] * 1.003 and rsi < 42:
        return {"direction": "bearish", "confidence": 0.82, "headline": "Breakdown – strong bearish momentum", "source": "technical"}
    return None

def get_event_reminders():
    """Only return reminders when event is close (1 day / hours / minutes)"""
    now = datetime.utcnow()
    reminders = []
    for event in HIGH_IMPACT_EVENTS:
        try:
            event_dt = datetime.strptime(f"{event['date']} {event['time']}", "%Y-%m-%d %H:%M")
            diff_hours = (event_dt - now).total_seconds() / 3600

            if 20 <= diff_hours <= 26:  # \~1 day before
                reminders.append(f"📅 REMINDER (1 day): {event['event']} tomorrow at {event['time']} UTC – {event['impact']} impact")
            elif 3 <= diff_hours <= 6:  # a few hours before
                reminders.append(f"⚠️ COMING SOON ({diff_hours:.1f}h): {event['event']} – Prepare for volatility")
            elif 0.25 <= diff_hours <= 1.5:  # minutes to 1.5 hours
                reminders.append(f"🚨 IMMINENT ({int(diff_hours*60)} min): {event['event']} starting soon!")
        except:
            continue
    return reminders

# Session state
if "positions" not in st.session_state:
    st.session_state.positions = {}
if "signals" not in st.session_state:
    st.session_state.signals = []
if "logs" not in st.session_state:
    st.session_state.logs = []
if "last_reminder_sent" not in st.session_state:
    st.session_state.last_reminder_sent = set()

# Auto refresh every 60 seconds
if auto_refresh:
    st.markdown('<meta http-equiv="refresh" content="60">', unsafe_allow_html=True)

run_analysis = st.button("🔄 Run Analysis Now", use_container_width=True) or auto_refresh

if run_analysis:
    st.session_state.logs = []
    with st.spinner("Scanning markets + checking event reminders..."):

        # ===== EVENT REMINDERS (only when close) =====
        reminders = get_event_reminders()
        for rem in reminders:
            if rem not in st.session_state.last_reminder_sent:
                st.session_state.logs.append(rem)
                if send_telegram_alerts:
                    send_telegram(rem)
                st.session_state.last_reminder_sent.add(rem)

        # ===== MARKET SCAN =====
        for symbol, info in WATCHLIST.items():
            name = info["name"]
            news = get_clear_news(info["keywords"])
            structure = get_structure(symbol)
            signal_data = news

            if signal_data is None:
                signal_data = get_strong_technical(structure)

            if signal_data is None:
                continue

            direction = signal_data["direction"]
            conf = signal_data["confidence"]
            headline = signal_data["headline"]
            source = signal_data.get("source", "news")

            if not structure:
                continue

            price = structure["price"]
            price_text = f"${price:,.2f}"
            entry_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

            st.session_state.logs.append(f"{name}: {direction.upper()} ({conf:.0%}) via {source} @ {price_text}")

            # EXIT
            if symbol in st.session_state.positions:
                pos = st.session_state.positions[symbol]
                if (pos["side"] == "long" and direction == "bearish") or (pos["side"] == "short" and direction == "bullish"):
                    if send_telegram_alerts:
                        send_telegram(f"🔴 <b>EXIT {pos['side'].upper()}</b> – {name}\nPrice: {price_text}\n{headline}")
                    del st.session_state.positions[symbol]
                    st.session_state.logs.append(f"→ EXIT sent for {name}")
                    continue

            # ENTRY
            if symbol not in st.session_state.positions:
                side = "long" if direction == "bullish" else "short"
                sl, tp = calc_sl_tp(structure, side)
                sl_text = f"${sl:,.2f}" if sl else "N/A"
                tp_text = f"${tp:,.2f}" if tp else "N/A"

                st.session_state.positions[symbol] = {
                    "side": side, "name": name, "entry_price": price_text,
                    "stop_loss": sl_text, "take_profit": tp_text,
                    "time": entry_time, "source": source, "headline": headline
                }

                if send_telegram_alerts:
                    msg = (f"🟢 <b>ENTRY {side.upper()}</b> – {name}\n"
                           f"Entry: {price_text}\nSL: {sl_text}\nTP: {tp_text}\n"
                           f"Time: {entry_time}\nSource: {source.upper()}\n\n{headline}")
                    send_telegram(msg)
                st.session_state.logs.append(f"→ ENTRY sent | {name} @ {price_text}")

    st.success("Scan complete")

# ====================== DISPLAY ======================
st.subheader("📊 Open Positions")
if st.session_state.positions:
    for s, p in st.session_state.positions.items():
        st.success(f"**{p['name']}** | {p['side'].upper()}\n"
                   f"Entry: {p['entry_price']} | SL: {p['stop_loss']} | TP: {p['take_profit']}\n"
                   f"Time: {p['time']}\n{p['headline']}")
else:
    st.info("No open positions")

st.subheader("📜 Recent Activity / Reminders")
if st.session_state.logs:
    for log in st.session_state.logs[-15:]:
        st.text(log)
else:
    st.write("Waiting for signals or upcoming event reminders...")

st.caption("Auto-refresh is active (every 60 seconds). High-impact events only alert when they are approaching.")