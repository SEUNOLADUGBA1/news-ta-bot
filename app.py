import streamlit as st
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from transformers import pipeline

st.set_page_config(page_title="News + TA Bot", page_icon="📈", layout="wide")
st.title("📈 News + Technical Analysis Bot")
st.caption("News-first + Strong Pure Technical Analysis | Gold & Silver Crosses included")

# ====================== SECRETS ======================
try:
    TELEGRAM_BOT_TOKEN = st.secrets["TELEGRAM_BOT_TOKEN"]
    TELEGRAM_CHAT_ID = st.secrets["TELEGRAM_CHAT_ID"]
    NEWS_API_KEY = st.secrets["NEWS_API_KEY"]
    st.sidebar.success("Keys loaded from Secrets")
except Exception:
    st.error("Secrets not found. Please add them in Streamlit Settings → Secrets")
    st.stop()

send_telegram_alerts = st.sidebar.checkbox("Send Telegram Alerts", value=True)

if st.sidebar.button("📨 Test Telegram"):
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": "✅ Bot test successful"},
            timeout=10
        )
        st.sidebar.success("Telegram OK" if r.status_code == 200 else r.text)
    except Exception as e:
        st.sidebar.error(str(e))

# Very clear news only
CONFIDENCE_THRESHOLD = 0.70

WATCHLIST = {
    # Crypto
    "BTC-USD": {"name": "Bitcoin", "keywords": ["Bitcoin", "BTC price"]},
    "ETH-USD": {"name": "Ethereum", "keywords": ["Ethereum", "ETH price"]},
    "SOL-USD": {"name": "Solana", "keywords": ["Solana", "SOL price"]},

    # Stocks
    "AAPL": {"name": "Apple", "keywords": ["Apple stock", "AAPL"]},
    "TSLA": {"name": "Tesla", "keywords": ["Tesla stock", "TSLA"]},
    "NVDA": {"name": "Nvidia", "keywords": ["Nvidia stock", "NVDA"]},

    # Major Forex (USD pairs)
    "EURUSD=X": {"name": "EUR/USD", "keywords": ["EURUSD", "EUR/USD", "euro dollar"]},
    "GBPUSD=X": {"name": "GBP/USD", "keywords": ["GBPUSD", "GBP/USD", "pound dollar"]},
    "AUDUSD=X": {"name": "AUD/USD", "keywords": ["AUDUSD", "AUD/USD", "aussie dollar"]},
    "USDJPY=X": {"name": "USD/JPY", "keywords": ["USDJPY", "USD/JPY"]},

    # Gold (XAU)
    "GC=F": {"name": "Gold (XAU/USD)", "keywords": ["gold price", "XAUUSD", "gold"]},
    # Gold crosses – using GC=F price as proxy + targeted keywords
    "GC=F_AUD": {"name": "XAU/AUD", "keywords": ["gold AUD", "gold Australia", "XAU AUD"], "price_symbol": "GC=F"},
    "GC=F_EUR": {"name": "XAU/EUR", "keywords": ["gold EUR", "gold euro", "XAU EUR"], "price_symbol": "GC=F"},
    "GC=F_GBP": {"name": "XAU/GBP", "keywords": ["gold GBP", "gold pound", "XAU GBP"], "price_symbol": "GC=F"},

    # Silver (XAG)
    "SI=F": {"name": "Silver (XAG/USD)", "keywords": ["silver price", "XAGUSD", "silver"]},
    "SI=F_EUR": {"name": "XAG/EUR", "keywords": ["silver EUR", "silver euro", "XAG EUR"], "price_symbol": "SI=F"},
    "SI=F_AUD": {"name": "XAG/AUD", "keywords": ["silver AUD", "silver Australia", "XAG AUD"], "price_symbol": "SI=F"},
    "SI=F_GBP": {"name": "XAG/GBP", "keywords": ["silver GBP", "silver pound", "XAG GBP"], "price_symbol": "SI=F"},

    # Energies
    "CL=F": {"name": "Crude Oil", "keywords": ["crude oil", "WTI", "oil price"]},
    "NG=F": {"name": "Natural Gas", "keywords": ["natural gas", "gas price"]},
}

@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert", truncation=True)

sentiment_model = load_sentiment_model()

def send_telegram(message: str) -> bool:
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"},
            timeout=10
        )
        return r.status_code == 200
    except:
        return False

def get_market_structure(symbol: str):
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

        # ATR
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        atr = float(tr.rolling(14).mean().iloc[-1])

        recent_high = float(high.tail(36).max())
        recent_low = float(low.tail(36).min())

        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = float(100 - (100 / (1 + rs.iloc[-1])))

        return {
            "price": price,
            "sma20": sma20,
            "sma50": sma50,
            "atr": atr,
            "recent_high": recent_high,
            "recent_low": recent_low,
            "rsi": rsi
        }
    except:
        return None

def calculate_sl_tp(structure, side: str):
    if not structure:
        return None, None
    price = structure["price"]
    atr = structure["atr"]
    high = structure["recent_high"]
    low = structure["recent_low"]

    if side == "long":
        sl = min(low, price - 1.7 * atr)
        risk = price - sl
        tp = price + risk * 1.8
    else:
        sl = max(high, price + 1.7 * atr)
        risk = sl - price
        tp = price - risk * 1.8
    return sl, tp

def get_very_clear_news(keywords: list):
    query = " OR ".join(keywords)
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 10,
        "apiKey": NEWS_API_KEY,
        "from": (datetime.utcnow() - timedelta(hours=18)).isoformat()
    }
    try:
        r = requests.get(url, params=params, timeout=12)
        articles = r.json().get("articles", [])
    except:
        return None

    best = None
    for art in articles:
        title = art.get("title") or ""
        if len(title) < 22:
            continue
        result = sentiment_model(title[:500])[0]
        label = result["label"].lower()
        score = result["score"]
        if label in ["positive", "negative"] and score >= CONFIDENCE_THRESHOLD:
            direction = "bullish" if label == "positive" else "bearish"
            if best is None or score > best["confidence"]:
                best = {"direction": direction, "confidence": score, "headline": title, "source": "news"}
    return best

def get_strong_technical_signal(structure) -> dict | None:
    """Pure technical signal when no news exists – strong conditions only"""
    if not structure:
        return None

    price = structure["price"]
    sma20 = structure["sma20"]
    sma50 = structure["sma50"]
    rsi = structure["rsi"]
    high = structure["recent_high"]
    low = structure["recent_low"]

    # Strong Bullish Technical
    if (price > sma20 > sma50 and
        rsi > 52 and rsi < 72 and
        price > (high + low) / 2):
        return {"direction": "bullish", "confidence": 0.75, "headline": "Strong bullish technical structure (price > SMA20 > SMA50 + healthy RSI)", "source": "technical"}

    # Strong Bearish Technical
    if (price < sma20 < sma50 and
        rsi < 48 and rsi > 28 and
        price < (high + low) / 2):
        return {"direction": "bearish", "confidence": 0.75, "headline": "Strong bearish technical structure (price < SMA20 < SMA50 + healthy RSI)", "source": "technical"}

    return None

# Session state
if "positions" not in st.session_state:
    st.session_state.positions = {}
if "signals" not in st.session_state:
    st.session_state.signals = []
if "logs" not in st.session_state:
    st.session_state.logs = []

if st.button("🔄 Run Analysis", use_container_width=True):
    st.session_state.logs = []
    with st.spinner("Scanning news + strong technicals..."):
        for key, info in WATCHLIST.items():
            name = info["name"]
            price_symbol = info.get("price_symbol", key.split("_")[0] if "_" in key else key)

            # 1. Try clear news first
            news = get_very_clear_news(info["keywords"])
            signal_data = news

            # 2. If no clear news → try strong pure technical
            structure = get_market_structure(price_symbol)
            if signal_data is None:
                signal_data = get_strong_technical_signal(structure)
                if signal_data:
                    st.session_state.logs.append(f"{name}: No clear news → Strong pure technical found")

            if signal_data is None:
                st.session_state.logs.append(f"{name}: No clear news and no strong technical")
                continue

            direction = signal_data["direction"]
            conf = signal_data["confidence"]
            headline = signal_data["headline"]
            source = signal_data.get("source", "news")

            if not structure:
                st.session_state.logs.append(f"{name}: Could not load price data")
                continue

            price = structure["price"]
            price_text = f"${price:,.2f}"
            entry_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

            st.session_state.logs.append(f"{name}: {direction.upper()} ({conf:.0%}) via {source} @ {price_text}")
            st.session_state.logs.append(f"   {headline[:80]}...")

            # EXIT
            if key in st.session_state.positions:
                pos = st.session_state.positions[key]
                if (pos["side"] == "long" and direction == "bearish") or (pos["side"] == "short" and direction == "bullish"):
                    signal = {
                        "type": "EXIT", "symbol": name, "side": pos["side"],
                        "price": price_text, "headline": headline, "time": entry_time, "confidence": conf
                    }
                    st.session_state.signals.insert(0, signal)
                    if send_telegram_alerts:
                        msg = (f"🔴 <b>EXIT {pos['side'].upper()}</b> – {name}\n"
                               f"Exit Price: {price_text}\nTime: {entry_time}\nReason: {headline}")
                        send_telegram(msg)
                    del st.session_state.positions[key]
                    st.session_state.logs.append(f"→ EXIT sent at {price_text}")
                    continue

            # ENTRY
            if key not in st.session_state.positions:
                side = "long" if direction == "bullish" else "short"
                sl, tp = calculate_sl_tp(structure, side)
                sl_text = f"${sl:,.2f}" if sl else "N/A"
                tp_text = f"${tp:,.2f}" if tp else "N/A"

                st.session_state.positions[key] = {
                    "side": side, "name": name, "entry_price": price_text,
                    "stop_loss": sl_text, "take_profit": tp_text,
                    "headline": headline, "confidence": conf, "time": entry_time, "source": source
                }

                signal = {
                    "type": "ENTRY", "symbol": name, "side": side,
                    "price": price_text, "stop_loss": sl_text, "take_profit": tp_text,
                    "headline": headline, "time": entry_time, "confidence": conf, "source": source
                }
                st.session_state.signals.insert(0, signal)

                if send_telegram_alerts:
                    msg = (f"🟢 <b>ENTRY {side.upper()}</b> – {name}\n\n"
                           f"Entry: {price_text}\nSL: {sl_text}\nTP: {tp_text}\n"
                           f"Time: {entry_time}\nSource: {source.upper()}\n\n"
                           f"{headline}\nConfidence: {conf:.0%}")
                    ok = send_telegram(msg)
                    st.session_state.logs.append(f"→ ENTRY sent | {price_text} | SL {sl_text} | TP {tp_text}" + (" ✓" if ok else " (TG failed)"))

    st.success("Analysis complete")

# ====================== DISPLAY ======================
st.subheader("📊 Open Positions")
if st.session_state.positions:
    for key, pos in st.session_state.positions.items():
        st.success(f"**{pos['name']}** | {pos['side'].upper()} | via {pos.get('source', 'news').upper()}\n\n"
                   f"Entry: {pos['entry_price']} | SL: {pos['stop_loss']} | TP: {pos['take_profit']}\n"
                   f"Time: {pos['time']}\n\n{pos['headline']}")
else:
    st.info("No open positions")

st.subheader("📜 Recent Signals")
if st.session_state.signals:
    for sig in st.session_state.signals[:12]:
        color = "green" if sig["type"] == "ENTRY" else "red"
        extra = f"\nSL: {sig.get('stop_loss')} | TP: {sig.get('take_profit')}" if sig["type"] == "ENTRY" else ""
        st.markdown(f":{color}[**{sig['type']} {sig['side'].upper()} – {sig['symbol']}**]  \n"
                    f"Price: {sig.get('price')}{extra}  \nTime: {sig.get('time')}  \n"
                    f"Source: {sig.get('source', 'news').upper()}  \n{sig['headline']}")
else:
    st.write("No signals yet")

st.subheader("🔍 Analysis Log")
if st.session_state.logs:
    for log in st.session_state.logs:
        st.text(log)
else:
    st.write("Click **Run Analysis** to start")