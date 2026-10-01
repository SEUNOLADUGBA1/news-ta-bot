import streamlit as st
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from transformers import pipeline
import time

st.set_page_config(page_title="News + TA Bot Pro", page_icon="📈", layout="wide")
st.title("📈 News + TA Bot – Economic Data + Jobs Reports")
st.caption("US NFP + UK / EU / Australia Jobs • Multi-stage reminders")

# ====================== SECRETS ======================
try:
    TELEGRAM_BOT_TOKEN = st.secrets["TELEGRAM_BOT_TOKEN"]
    TELEGRAM_CHAT_ID = st.secrets["TELEGRAM_CHAT_ID"]
    NEWS_API_KEY = st.secrets["NEWS_API_KEY"]
    st.sidebar.success("Keys loaded")
except:
    st.error("Secrets missing")
    st.stop()

send_telegram_alerts = st.sidebar.checkbox("Send Telegram Alerts", value=True)
auto_refresh = st.sidebar.checkbox("Auto Refresh (60 sec)", value=True)

if st.sidebar.button("📨 Test Telegram"):
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": "✅ Bot working"},
            timeout=8
        )
        st.sidebar.success("OK" if r.status_code == 200 else r.text)
    except Exception as e:
        st.sidebar.error(str(e))

CONFIDENCE_THRESHOLD = 0.47

WATCHLIST = {
    "BTC-USD": {"name": "Bitcoin", "keywords": ["Bitcoin", "BTC"]},
    "ETH-USD": {"name": "Ethereum", "keywords": ["Ethereum", "ETH"]},
    "EURUSD=X": {"name": "EUR/USD", "keywords": ["EURUSD", "EUR/USD"]},
    "GBPUSD=X": {"name": "GBP/USD", "keywords": ["GBPUSD", "GBP/USD"]},
    "AUDUSD=X": {"name": "AUD/USD", "keywords": ["AUDUSD", "AUD/USD"]},
    "USDJPY=X": {"name": "USD/JPY", "keywords": ["USDJPY", "USD/JPY"]},
    "GC=F": {"name": "Gold", "keywords": ["gold", "XAUUSD"]},
    "SI=F": {"name": "Silver", "keywords": ["silver", "XAGUSD"]},
    "CL=F": {"name": "Crude Oil", "keywords": ["oil", "crude", "WTI"]},
}

# ====================== HIGH IMPACT EVENTS ======================
# Includes tomorrow's US NFP + UK / EU / Australia jobs reports
HIGH_IMPACT_EVENTS = [
    # Tomorrow's US NFP (adjust exact time if needed)
    {
        "date": "2026-10-02",
        "time": "12:30",
        "event": "US Non-Farm Payrolls (NFP)",
        "impact": "Very High",
        "outcomes": "Strong NFP → USD bullish, Gold bearish | Weak NFP → USD bearish, Gold bullish"
    },
    # Other major jobs reports
    {
        "date": "2026-10-15",
        "time": "06:00",
        "event": "UK Unemployment Rate / Employment Change",
        "impact": "High",
        "outcomes": "Lower unemployment → GBP bullish | Higher unemployment → GBP bearish"
    },
    {
        "date": "2026-10-16",
        "time": "09:00",
        "event": "Eurozone Unemployment Rate",
        "impact": "High",
        "outcomes": "Lower unemployment → EUR supportive | Higher → EUR weaker"
    },
    {
        "date": "2026-10-16",
        "time": "01:30",
        "event": "Australia Employment Change / Unemployment Rate",
        "impact": "High",
        "outcomes": "Strong jobs → AUD bullish | Weak jobs → AUD bearish"
    },
    # Other important events kept
    {
        "date": "2026-10-03",
        "time": "14:00",
        "event": "ISM Services PMI",
        "impact": "High",
        "outcomes": "Strong services → USD supportive"
    },
    {
        "date": "2026-10-14",
        "time": "12:30",
        "event": "US CPI",
        "impact": "Very High",
        "outcomes": "Hot CPI → USD up | Cool CPI → USD down + Gold up"
    },
    {
        "date": "2026-10-28",
        "time": "18:00",
        "event": "FOMC Decision + Press Conference",
        "impact": "Very High",
        "outcomes": "Hawkish → USD up | Dovish → USD down + Gold up"
    },
]

ECONOMIC_KEYWORDS = [
    "NFP", "Nonfarm", "Non-Farm", "Payrolls", "Unemployment", "Employment Change",
    "ISM", "PMI", "CPI", "PPI", "Core PCE", "GDP", "Retail Sales",
    "Jobless Claims", "FOMC", "Fed Minutes", "Interest Rate", "ECB", "BoE", "RBA",
    "Consumer Confidence", "Michigan", "Building Permits", "Housing Starts"
]

@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert", truncation=True)

sentiment_model = load_model()

def send_telegram(msg):
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"},
            timeout=8
        )
        return True
    except:
        return False

def get_structure(symbol):
    try:
        df = yf.download(symbol, period="7d", interval="1h", progress=False)
        if df.empty or len(df) < 25:
            return None
        df = df.dropna()
        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        price = float(close.iloc[-1])
        sma20 = float(close.rolling(20).mean().iloc[-1])
        tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
        atr = float(tr.rolling(14).mean().iloc[-1])
        recent_high = float(high.tail(24).max())
        recent_low = float(low.tail(24).min())
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = float(100 - (100 / (1 + rs.iloc[-1])))
        return {"price": price, "sma20": sma20, "atr": atr, "recent_high": recent_high, "recent_low": recent_low, "rsi": rsi}
    except:
        return None

def calc_sl_tp(structure, side):
    if not structure:
        return None, None
    price = structure["price"]
    atr = structure["atr"]
    if side == "long":
        sl = min(structure["recent_low"], price - 1.5 * atr)
        tp = price + (price - sl) * 1.6
    else:
        sl = max(structure["recent_high"], price + 1.5 * atr)
        tp = price - (sl - price) * 1.6
    return sl, tp

def get_economic_news():
    query = " OR ".join(["NFP", "Nonfarm", "Payrolls", "Unemployment", "Employment", "ISM", "CPI"])
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 12,
        "apiKey": NEWS_API_KEY,
        "from": (datetime.utcnow() - timedelta(hours=10)).isoformat()
    }
    try:
        r = requests.get("https://newsapi.org/v2/everything", params=params, timeout=10)
        articles = r.json().get("articles", [])
    except:
        return None

    best = None
    for art in articles:
        title = art.get("title") or ""
        if len(title) < 15:
            continue
        res = sentiment_model(title[:450])[0]
        label = res["label"].lower()
        score = res["score"]
        if label in ["positive", "negative"] and score >= CONFIDENCE_THRESHOLD:
            direction = "bullish" if label == "positive" else "bearish"
            if best is None or score > best["confidence"]:
                best = {"direction": direction, "confidence": score, "headline": title, "source": "economic"}
    return best

def get_strong_technical(structure):
    if not structure:
        return None
    price = structure["price"]
    sma20 = structure["sma20"]
    rsi = structure["rsi"]
    if price > sma20 and 48 < rsi < 72:
        return {"direction": "bullish", "confidence": 0.67, "headline": "Bullish technical", "source": "technical"}
    if price < sma20 and 28 < rsi < 52:
        return {"direction": "bearish", "confidence": 0.67, "headline": "Bearish technical", "source": "technical"}
    return None

def get_smart_reminders():
    now = datetime.utcnow()
    reminders = []
    for event in HIGH_IMPACT_EVENTS:
        try:
            event_dt = datetime.strptime(f"{event['date']} {event['time']}", "%Y-%m-%d %H:%M")
            diff_sec = (event_dt - now).total_seconds()
            diff_hours = diff_sec / 3600
            diff_min = diff_sec / 60
            name = event["event"]
            impact = event.get("impact", "High")
            outcomes = event.get("outcomes", "")

            # 1 day before – full message with possible outcomes
            if 18 <= diff_hours <= 30:
                reminders.append(f"📅 1 DAY BEFORE: {name} ({impact})\nPossible outcomes:\n{outcomes}")

            # Later stages – short only
            elif 11 <= diff_hours <= 13:
                reminders.append(f"12h → {name}")
            elif 5.5 <= diff_hours <= 6.5:
                reminders.append(f"6h → {name}")
            elif 2.7 <= diff_hours <= 3.3:
                reminders.append(f"3h → {name}")
            elif 1.7 <= diff_hours <= 2.3:
                reminders.append(f"2h → {name}")
            elif 0.8 <= diff_hours <= 1.2:
                reminders.append(f"1h → {name}")
            elif 25 <= diff_min <= 40:
                reminders.append(f"30 min → {name}")
            elif 8 <= diff_min <= 15:
                reminders.append(f"10 min → {name}")
            elif 4 <= diff_min <= 7:
                reminders.append(f"5 min → {name}")
            elif 0.3 <= diff_min <= 2:
                reminders.append(f"1 min → {name} almost live!")
        except:
            continue
    return reminders

# Session state
if "positions" not in st.session_state:
    st.session_state.positions = {}
if "logs" not in st.session_state:
    st.session_state.logs = []
if "sent_reminders" not in st.session_state:
    st.session_state.sent_reminders = set()

if auto_refresh:
    st.markdown('<meta http-equiv="refresh" content="60">', unsafe_allow_html=True)

run = st.button("🔄 Run Scan", use_container_width=True) or auto_refresh

if run:
    st.session_state.logs = []
    with st.spinner("Scanning + checking NFP & jobs reminders..."):

        # Reminders
        for rem in get_smart_reminders():
            key = rem[:50]
            if key not in st.session_state.sent_reminders:
                st.session_state.logs.append(rem)
                if send_telegram_alerts:
                    send_telegram(rem)
                st.session_state.sent_reminders.add(key)

        # Economic news (NFP etc.)
        econ = get_economic_news()
        if econ:
            st.session_state.logs.append(f"NEWS: {econ['direction'].upper()} ({econ['confidence']:.0%}) → {econ['headline'][:75]}")

        for symbol, info in WATCHLIST.items():
            name = info["name"]
            structure = get_structure(symbol)
            signal = econ if econ else get_strong_technical(structure)

            if not signal or not structure:
                continue

            direction = signal["direction"]
            conf = signal["confidence"]
            headline = signal["headline"]
            price = structure["price"]
            price_text = f"${price:,.2f}"

            st.session_state.logs.append(f"{name}: {direction.upper()} ({conf:.0%}) @ {price_text}")

            # EXIT
            if symbol in st.session_state.positions:
                pos = st.session_state.positions[symbol]
                if (pos["side"] == "long" and direction == "bearish") or (pos["side"] == "short" and direction == "bullish"):
                    if send_telegram_alerts:
                        send_telegram(f"🔴 EXIT {pos['side'].upper()} – {name}\n{price_text}\n{headline}")
                    del st.session_state.positions[symbol]
                    continue

            # ENTRY
            if symbol not in st.session_state.positions:
                side = "long" if direction == "bullish" else "short"
                sl, tp = calc_sl_tp(structure, side)
                sl_text = f"${sl:,.2f}" if sl else "N/A"
                tp_text = f"${tp:,.2f}" if tp else "N/A"

                st.session_state.positions[symbol] = {
                    "side": side, "name": name, "entry_price": price_text,
                    "stop_loss": sl_text, "take_profit": tp_text, "headline": headline
                }
                if send_telegram_alerts:
                    send_telegram(f"🟢 ENTRY {side.upper()} – {name}\nEntry: {price_text}\nSL: {sl_text}\nTP: {tp_text}\n{headline}")

    st.success("Scan complete")

# Display
st.subheader("📊 Open Positions")
if st.session_state.positions:
    for s, p in st.session_state.positions.items():
        st.success(f"**{p['name']}** {p['side'].upper()} | {p['entry_price']} | SL {p['stop_loss']} | TP {p['take_profit']}\n{p['headline']}")
else:
    st.info("No open positions")

st.subheader("🔍 Log / Reminders")
for log in st.session_state.logs[-12:]:
    st.text(log)

st.caption("US NFP + UK / EU / Australia jobs reports included • Reminders from 1 day → 1 minute")