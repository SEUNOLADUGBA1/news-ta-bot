import streamlit as st
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from transformers import pipeline
import pytz

st.set_page_config(page_title="News + TA Bot + Calendar", page_icon="📈", layout="wide")
st.title("📈 News + Technical Analysis Bot")
st.caption("Strong Gold detection + High-Impact Events Calendar + Forecasts")

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

# ====================== HIGH IMPACT CALENDAR (from your picture) ======================
# Times are approximate UK time converted for simplicity
HIGH_IMPACT_EVENTS = [
    {"date": "2026-09-30", "time": "02:30", "event": "Australia CPI (Aug)", "impact": "High", "affects": ["AUD/USD", "Gold", "XAU/AUD"]},
    {"date": "2026-10-02", "time": "12:30", "event": "US NFP (Sep)", "impact": "Very High", "affects": ["USD pairs", "Gold", "Silver", "Oil"]},
    {"date": "2026-10-02", "time": "10:00", "event": "EU Flash CPI (Sep)", "impact": "High", "affects": ["EUR/USD", "Gold"]},
    {"date": "2026-10-14", "time": "13:30", "event": "US CPI (Sep)", "impact": "Very High", "affects": ["USD pairs", "Gold", "Silver"]},
    {"date": "2026-10-15", "time": "13:30", "event": "US PPI (Sep)", "impact": "High", "affects": ["USD pairs", "Gold"]},
    {"date": "2026-10-21", "time": "07:00", "event": "UK CPI + PPI", "impact": "High", "affects": ["GBP/USD", "Gold"]},
    {"date": "2026-10-28", "time": "18:00", "event": "FOMC Decision + Presser", "impact": "Very High", "affects": ["All USD pairs", "Gold", "Silver", "Bitcoin"]},
    {"date": "2026-10-29", "time": "13:15", "event": "ECB Decision + Lagarde", "impact": "Very High", "affects": ["EUR/USD", "Gold"]},
    {"date": "2026-11-05", "time": "12:00", "event": "BoE Decision", "impact": "High", "affects": ["GBP/USD", "Gold"]},
    {"date": "2026-11-06", "time": "12:30", "event": "US NFP (Oct)", "impact": "Very High", "affects": ["USD pairs", "Gold", "Silver"]},
    {"date": "2026-11-10", "time": "13:30", "event": "US CPI (Oct)", "impact": "Very High", "affects": ["USD pairs", "Gold"]},
    {"date": "2026-12-09", "time": "18:00", "event": "FOMC Decision + Presser", "impact": "Very High", "affects": ["All markets"]},
]

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
    "GC=F": {"name": "Gold (XAU/USD)", "keywords": ["gold price", "XAUUSD", "gold", "bullion"]},
    "SI=F": {"name": "Silver (XAG/USD)", "keywords": ["silver price", "XAG", "silver"]},
    "CL=F": {"name": "Crude Oil", "keywords": ["crude oil", "WTI", "oil price"]},
    "NG=F": {"name": "Natural Gas", "keywords": ["natural gas", "gas price"]},
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
        recent_high = float(high.tail(40).max())
        recent_low = float(low.tail(40).min())

        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = float(100 - (100 / (1 + gain/loss)).iloc[-1])

        return {"price": price, "sma20": sma20, "sma50": sma50, "atr": atr,
                "recent_high": recent_high, "recent_low": recent_low, "rsi": rsi}
    except:
        return None

def calc_sl_tp(structure, side):
    if not structure:
        return None, None
    price = structure["price"]
    atr = structure["atr"]
    if side == "long":
        sl = min(structure["recent_low"], price - 1.6 * atr)
        tp = price + (price - sl) * 1.8
    else:
        sl = max(structure["recent_high"], price + 1.6 * atr)
        tp = price - (sl - price) * 1.8
    return sl, tp

def get_clear_news(keywords):
    query = " OR ".join(keywords)
    params = {
        "q": query, "language": "en", "sortBy": "publishedAt",
        "pageSize": 10, "apiKey": NEWS_API_KEY,
        "from": (datetime.utcnow() - timedelta(hours=20)).isoformat()
    }
    try:
        r = requests.get("https://newsapi.org/v2/everything", params=params, timeout=12)
        articles = r.json().get("articles", [])
    except:
        return None

    best = None
    for art in articles:
        title = art.get("title") or ""
        if len(title) < 20:
            continue
        res = sentiment_model(title[:500])[0]
        label = res["label"].lower()
        score = res["score"]
        if label in ["positive", "negative"] and score >= CONFIDENCE_THRESHOLD:
            direction = "bullish" if label == "positive" else "bearish"
            if best is None or score > best["confidence"]:
                best = {"direction": direction, "confidence": score, "headline": title, "source": "news"}
    return best

def strong_technical(structure, is_gold_or_silver=False):
    """Stronger pure technical – especially sensitive for Gold/Silver"""
    if not structure:
        return None
    price = structure["price"]
    sma20 = structure["sma20"]
    sma50 = structure["sma50"]
    rsi = structure["rsi"]
    high = structure["recent_high"]
    low = structure["recent_low"]

    # Stronger conditions for Gold & Silver
    if is_gold_or_silver:
        # Bearish breakdown
        if price < sma20 and price < sma50 and rsi < 45:
            return {"direction": "bearish", "confidence": 0.78,
                    "headline": "Strong bearish technical on Gold/Silver (below SMAs + weak RSI)", "source": "technical"}
        # Bullish recovery
        if price > sma20 and rsi > 50 and rsi < 70:
            return {"direction": "bullish", "confidence": 0.75,
                    "headline": "Strong bullish technical recovery on Gold/Silver", "source": "technical"}

    # General strong technical
    if price > sma20 > sma50 and 52 < rsi < 70:
        return {"direction": "bullish", "confidence": 0.74,
                "headline": "Strong bullish technical structure", "source": "technical"}
    if price < sma20 < sma50 and 30 < rsi < 48:
        return {"direction": "bearish", "confidence": 0.74,
                "headline": "Strong bearish technical structure", "source": "technical"}
    return None

def check_upcoming_events():
    """Return events happening in the next 48 hours"""
    now = datetime.utcnow()
    upcoming = []
    for ev in HIGH_IMPACT_EVENTS:
        try:
            ev_dt = datetime.strptime(f"{ev['date']} {ev['time']}", "%Y-%m-%d %H:%M")
            diff = (ev_dt - now).total_seconds() / 3600
            if -2 < diff < 48:  # past 2h to next 48h
                upcoming.append({**ev, "hours_away": round(diff, 1)})
        except:
            continue
    return sorted(upcoming, key=lambda x: x["hours_away"])

# ====================== SESSION ======================
if "positions" not in st.session_state:
    st.session_state.positions = {}
if "signals" not in st.session_state:
    st.session_state.signals = []
if "logs" not in st.session_state:
    st.session_state.logs = []

# Show upcoming events
st.subheader("📅 Upcoming High-Impact Events")
upcoming = check_upcoming_events()
if upcoming:
    for ev in upcoming:
        hours = ev["hours_away"]
        status = "🔴 LIVE / JUST PASSED" if hours < 1 else f"in {hours} hours"
        st.warning(f"**{ev['event']}** ({ev['impact']}) — {status}\nAffects: {', '.join(ev['affects'])}")
else:
    st.info("No high-impact events in the next 48 hours")

if st.button("🔄 Run Full Analysis + Forecasts", use_container_width=True):
    st.session_state.logs = []
    with st.spinner("Running news + strong technicals + calendar forecasts..."):

        # ===== CALENDAR FORECASTS =====
        for ev in upcoming:
            if 0 < ev["hours_away"] < 12:
                forecast_msg = (
                    f"⚠️ <b>HIGH IMPACT EVENT SOON</b>\n\n"
                    f"Event: {ev['event']}\n"
                    f"Time: in {ev['hours_away']} hours\n"
                    f"Impact: {ev['impact']}\n"
                    f"Affects: {', '.join(ev['affects'])}\n\n"
                    f"Expect increased volatility on Gold, USD pairs and related assets."
                )
                st.session_state.logs.append(f"FORECAST: {ev['event']} in {ev['hours_away']}h")
                if send_telegram_alerts:
                    send_telegram(forecast_msg)

        # ===== NORMAL SCAN =====
        for symbol, info in WATCHLIST.items():
            name = info["name"]
            is_metal = "Gold" in name or "Silver" in name

            news = get_clear_news(info["keywords"])
            structure = get_structure(symbol)

            signal_data = news
            if signal_data is None:
                signal_data = strong_technical(structure, is_gold_or_silver=is_metal)
                if signal_data:
                    st.session_state.logs.append(f"{name}: Pure strong technical triggered")

            if not signal_data:
                st.session_state.logs.append(f"{name}: No clear news + no strong technical")
                continue

            if not structure:
                continue

            direction = signal_data["direction"]
            conf = signal_data["confidence"]
            headline = signal_data["headline"]
            source = signal_data.get("source", "news")
            price = structure["price"]
            price_text = f"${price:,.2f}"
            entry_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

            st.session_state.logs.append(f"{name}: {direction.upper()} ({conf:.0%}) via {source} @ {price_text}")

            # EXIT
            if symbol in st.session_state.positions:
                pos = st.session_state.positions[symbol]
                if (pos["side"] == "long" and direction == "bearish") or (pos["side"] == "short" and direction == "bullish"):
                    if send_telegram_alerts:
                        send_telegram(f"🔴 <b>EXIT {pos['side'].upper()}</b> – {name}\nPrice: {price_text}\nTime: {entry_time}\n{headline}")
                    del st.session_state.positions[symbol]
                    st.session_state.logs.append(f"→ EXIT {name}")
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
                    "time": entry_time, "headline": headline, "source": source
                }
                st.session_state.signals.insert(0, {
                    "type": "ENTRY", "symbol": name, "side": side,
                    "price": price_text, "sl": sl_text, "tp": tp_text,
                    "time": entry_time, "headline": headline, "source": source
                })

                if send_telegram_alerts:
                    msg = (f"🟢 <b>ENTRY {side.upper()}</b> – {name}\n\n"
                           f"Entry: {price_text}\nSL: {sl_text}\nTP: {tp_text}\n"
                           f"Time: {entry_time}\nSource: {source.upper()}\n\n{headline}")
                    send_telegram(msg)
                    st.session_state.logs.append(f"→ ENTRY {name} sent")

    st.success("Full analysis + forecasts completed")

# ====================== DISPLAY ======================
st.subheader("📊 Open Positions")
if st.session_state.positions:
    for s, p in st.session_state.positions.items():
        st.success(f"**{p['name']}** | {p['side'].upper()} | {p.get('source','')}\n"
                   f"Entry: {p['entry_price']} | SL: {p['stop_loss']} | TP: {p['take_profit']}\n"
                   f"{p['time']}\n{p['headline']}")
else:
    st.info("No open positions")

st.subheader("📜 Recent Signals")
if st.session_state.signals:
    for sig in st.session_state.signals[:10]:
        color = "green" if sig["type"] == "ENTRY" else "red"
        st.markdown(f":{color}[**{sig['type']} {sig['side'].upper()} – {sig['symbol']}**]  \n"
                    f"{sig.get('price')} | SL: {sig.get('sl')} | TP: {sig.get('tp')}  \n"
                    f"{sig.get('time')} | {sig.get('source')}  \n{sig.get('headline')}")
else:
    st.write("No signals yet")

st.subheader("🔍 Log")
if st.session_state.logs:
    for log in st.session_state.logs:
        st.text(log)
else:
    st.write("Click the button to run")