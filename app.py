import streamlit as st
import requests
import yfinance as yf
from datetime import datetime, timedelta
from transformers import pipeline

st.set_page_config(page_title="News + TA Bot", page_icon="📈", layout="wide")
st.title("📈 News + Technical Analysis Bot")
st.caption("Stricter News + Improved Entry / SL / TP")

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
    "GC=F": {
        "name": "Gold",
        "keywords": ["gold price", "XAUUSD", "gold rises", "gold falls", "bullion"],
        "must_contain": ["gold", "xau", "bullion"]
    },
    "SI=F": {
        "name": "Silver",
        "keywords": ["silver price", "XAGUSD", "silver rises", "silver falls"],
        "must_contain": ["silver", "xag"]
    },
    "CL=F": {
        "name": "Crude Oil",
        "keywords": ["crude oil", "WTI", "oil price", "Brent"],
        "must_contain": ["oil", "wti", "brent", "crude"]
    },
    "BTC-USD": {
        "name": "Bitcoin",
        "keywords": ["Bitcoin", "BTC price", "Bitcoin rises", "Bitcoin falls"],
        "must_contain": ["bitcoin", "btc"]
    },
    "EURUSD=X": {
        "name": "EUR/USD",
        "keywords": ["EURUSD", "EUR/USD", "euro dollar"],
        "must_contain": ["eur", "euro"]
    },
    "GBPUSD=X": {
        "name": "GBP/USD",
        "keywords": ["GBPUSD", "GBP/USD", "pound dollar"],
        "must_contain": ["gbp", "pound", "sterling"]
    },
    "AUDUSD=X": {
        "name": "AUD/USD",
        "keywords": ["AUDUSD", "AUD/USD", "aussie"],
        "must_contain": ["aud", "aussie", "australia"]
    },
    "USDJPY=X": {
        "name": "USD/JPY",
        "keywords": ["USDJPY", "USD/JPY", "yen"],
        "must_contain": ["jpy", "yen", "japan"]
    },
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

def get_price_data(symbol):
    try:
        df = yf.download(symbol, period="5d", interval="1h", progress=False)
        if not df.empty and len(df) >= 15:
            df = df.dropna()
            close = df["Close"]
            high = df["High"]
            low = df["Low"]
            price = float(close.iloc[-1])
            sma20 = float(close.rolling(min(20, len(close))).mean().iloc[-1])
            atr = float((high - low).rolling(14).mean().iloc[-1])
            change = ((price - float(close.iloc[0])) / float(close.iloc[0])) * 100

            if price > sma20 * 1.001 and change > 0.3:
                direction, confidence = "bullish", 0.75
            elif price < sma20 * 0.999 and change < -0.3:
                direction, confidence = "bearish", 0.75
            elif price > sma20:
                direction, confidence = "bullish", 0.60
            elif price < sma20:
                direction, confidence = "bearish", 0.60
            else:
                direction, confidence = "neutral", 0.50

            reason = f"Price {price:.2f} | SMA {sma20:.2f} | Chg {change:.2f}%"
            return price, direction, reason, atr, confidence
    except:
        pass

    # Fallback
    try:
        ticker = yf.Ticker(symbol)
        price = float(ticker.fast_info["lastPrice"])
        atr = price * 0.012
        try:
            prev = float(ticker.fast_info["previousClose"])
            change = ((price - prev) / prev) * 100
            if change > 0.35:
                direction, confidence = "bullish", 0.65
            elif change < -0.35:
                direction, confidence = "bearish", 0.65
            else:
                direction, confidence = "neutral", 0.50
            reason = f"Price {price:.2f} | Chg {change:.2f}%"
        except:
            direction, confidence = "neutral", 0.50
            reason = f"Price {price:.2f}"
        return price, direction, reason, atr, confidence
    except Exception as e:
        return None, "error", str(e), None, 0

def calculate_sl_tp(price, direction, atr):
    """Improved SL/TP with cleaner risk-reward"""
    if price is None or atr is None or atr == 0:
        # fallback percentage
        if direction == "bullish":
            return price * 0.985, price * 1.025
        else:
            return price * 1.015, price * 0.975

    if direction == "bullish":
        sl = price - (1.4 * atr)
        tp = price + (2.1 * atr)
    else:  # bearish
        sl = price + (1.4 * atr)
        tp = price - (2.1 * atr)
    return sl, tp

def is_relevant(title, must_contain):
    """Stricter relevance check"""
    title_lower = title.lower()
    return any(word in title_lower for word in must_contain)

def get_news(info):
    query = " OR ".join(info["keywords"])
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 10,
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
        if len(title) < 15:
            continue
        # Stricter filter
        if not is_relevant(title, info["must_contain"]):
            continue

        try:
            result = sentiment_model(title[:400])[0]
            label = result["label"].lower()
            score = result["score"]
            if label in ["positive", "negative"] and score >= 0.58:
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

    with st.spinner("Running stricter analysis..."):
        for symbol, info in WATCHLIST.items():
            name = info["name"]
            st.session_state.logs.append(f"——— {name} ———")

            price, ta_direction, ta_reason, atr, ta_conf = get_price_data(symbol)
            if price is None:
                st.session_state.logs.append(f"ERROR: {ta_reason}")
                continue

            st.session_state.logs.append(f"Technical: {ta_reason}")
            st.session_state.logs.append(f"TA: {ta_direction.upper()} ({ta_conf:.0%})")

            news = get_news(info)

            if news:
                final_direction = news["direction"]
                confidence = news["confidence"]
                source = "News + Technical"
                headline = news["headline"]
                st.session_state.logs.append(f"News: {final_direction.upper()} ({confidence:.0%})")
                st.session_state.logs.append(f"Headline: {headline[:70]}...")
            else:
                final_direction = ta_direction
                confidence = ta_conf
                source = "Technical only"
                headline = "No relevant news found"
                st.session_state.logs.append("News: None (strict filter)")

            if final_direction in ["bullish", "bearish"]:
                sl, tp = calculate_sl_tp(price, final_direction, atr)
                price_text = f"${price:,.2f}"
                sl_text = f"${sl:,.2f}"
                tp_text = f"${tp:,.2f}"

                # Risk-Reward check
                risk = abs(price - sl)
                reward = abs(tp - price)
                rr = reward / risk if risk > 0 else 0

                signal = {
                    "name": name,
                    "direction": final_direction,
                    "price": price_text,
                    "sl": sl_text,
                    "tp": tp_text,
                    "confidence": confidence,
                    "ta": ta_reason,
                    "headline": headline,
                    "source": source,
                    "rr": rr,
                    "time": datetime.utcnow().strftime("%H:%M UTC")
                }
                st.session_state.signals.append(signal)
                st.session_state.logs.append(
                    f"✅ ENTRY {final_direction.upper()} | Conf {confidence:.0%} | "
                    f"SL {sl_text} | TP {tp_text} | RR 1:{rr:.1f}"
                )

                if send_telegram_alerts:
                    emoji = "🟢" if final_direction == "bullish" else "🔴"
                    msg = (
                        f"{emoji} <b>ENTRY {final_direction.upper()}</b> – {name}\n\n"
                        f"Entry: {price_text}\n"
                        f"Stop Loss: {sl_text}\n"
                        f"Take Profit: {tp_text}\n"
                        f"Risk:Reward ≈ 1:{rr:.1f}\n"
                        f"Confidence: {confidence:.0%}\n"
                        f"Technical: {ta_reason}\n"
                        f"Source: {source}\n"
                        f"Time: {signal['time']}\n\n"
                        f"{headline}"
                    )
                    ok = send_telegram(msg)
                    st.session_state.logs.append("Telegram: Sent" if ok else "Telegram: Failed")
            else:
                st.session_state.logs.append("Result: Neutral – no entry")

    st.success("Analysis complete")

# ====================== DISPLAY ======================
st.subheader("📊 Entry Signals")
if st.session_state.signals:
    for s in st.session_state.signals:
        color = "green" if s["direction"] == "bullish" else "red"
        st.markdown(
            f":{color}[**ENTRY {s['direction'].upper()} – {s['name']}**]\n\n"
            f"**Entry:** {s['price']}\n"
            f"**Stop Loss:** {s['sl']}\n"
            f"**Take Profit:** {s['tp']}\n"
            f"**Risk:Reward:** 1:{s['rr']:.1f}\n"
            f"**Confidence:** {s['confidence']:.0%}\n"
            f"**Technical:** {s['ta']}\n"
            f"**Source:** {s['source']}\n"
            f"**Time:** {s['time']}\n"
            f"{s['headline']}"
        )
else:
    st.info("No clear Entry signals right now.")

st.subheader("🔍 Log")
for line in st.session_state.logs:
    st.text(line)