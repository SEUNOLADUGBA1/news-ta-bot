import streamlit as st
import requests
import yfinance as yf
from datetime import datetime, timedelta
from transformers import pipeline

st.set_page_config(page_title="News + TA Bot", page_icon="📈", layout="wide")
st.title("📈 News + Technical Analysis Bot")
st.caption("Auto market hours • Weekend = open markets only • 1:3 RR")

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

# Full watchlist
FULL_WATCHLIST = {
    "BTC-USD": {"name": "Bitcoin", "keywords": ["Bitcoin", "BTC price"], "must_contain": ["bitcoin", "btc"], "open_weekend": True},
    "GC=F": {"name": "Gold", "keywords": ["gold price", "XAUUSD", "gold"], "must_contain": ["gold", "xau", "bullion"], "open_weekend": False},
    "SI=F": {"name": "Silver", "keywords": ["silver price", "XAGUSD", "silver"], "must_contain": ["silver", "xag"], "open_weekend": False},
    "CL=F": {"name": "Crude Oil", "keywords": ["crude oil", "WTI", "oil price"], "must_contain": ["oil", "wti", "brent", "crude"], "open_weekend": False},
    "EURUSD=X": {"name": "EUR/USD", "keywords": ["EURUSD", "EUR/USD"], "must_contain": ["eur", "euro"], "open_weekend": False},
    "GBPUSD=X": {"name": "GBP/USD", "keywords": ["GBPUSD", "GBP/USD"], "must_contain": ["gbp", "pound"], "open_weekend": False},
    "AUDUSD=X": {"name": "AUD/USD", "keywords": ["AUDUSD", "AUD/USD"], "must_contain": ["aud", "aussie"], "open_weekend": False},
    "USDJPY=X": {"name": "USD/JPY", "keywords": ["USDJPY", "USD/JPY"], "must_contain": ["jpy", "yen"], "open_weekend": False},
}

def is_weekend_now():
    """Approximate: Saturday + Sunday UTC = weekend (forex/metals mostly closed)"""
    now = datetime.utcnow()
    # weekday(): Mon=0 ... Sat=5, Sun=6
    return now.weekday() >= 5

def get_active_watchlist():
    if is_weekend_now():
        # Only markets that stay open
        return {k: v for k, v in FULL_WATCHLIST.items() if v.get("open_weekend", False)}
    return FULL_WATCHLIST

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
        if not df.empty and len(df) >= 20:
            df = df.dropna()
            close = df["Close"]
            high = df["High"]
            low = df["Low"]
            price = float(close.iloc[-1])
            sma20 = float(close.rolling(20).mean().iloc[-1])
            atr = float((high - low).rolling(14).mean().iloc[-1])
            change = ((price - float(close.iloc[0])) / float(close.iloc[0])) * 100

            recent = df.tail(12)
            up = (recent["Close"] > recent["Open"]).sum()
            buyers = int((up / len(recent)) * 100)
            sellers = 100 - buyers

            if price > sma20 * 1.001 and change > 0.3:
                direction, conf = "bullish", 0.75
            elif price < sma20 * 0.999 and change < -0.3:
                direction, conf = "bearish", 0.75
            elif price > sma20:
                direction, conf = "bullish", 0.62
            elif price < sma20:
                direction, conf = "bearish", 0.62
            else:
                direction, conf = "neutral", 0.50

            reason = f"Price {price:.2f} | SMA {sma20:.2f} | Chg {change:.2f}%"
            return price, direction, reason, atr, conf, buyers, sellers
    except:
        pass

    try:
        ticker = yf.Ticker(symbol)
        price = float(ticker.fast_info["lastPrice"])
        atr = price * 0.01
        try:
            prev = float(ticker.fast_info["previousClose"])
            change = ((price - prev) / prev) * 100
            if change > 0.35:
                direction, conf = "bullish", 0.65
            elif change < -0.35:
                direction, conf = "bearish", 0.65
            else:
                direction, conf = "neutral", 0.50
            reason = f"Price {price:.2f} | Chg {change:.2f}%"
        except:
            direction, conf = "neutral", 0.50
            reason = f"Price {price:.2f}"
        buyers = 55 if direction == "bullish" else 45 if direction == "bearish" else 50
        return price, direction, reason, atr, conf, buyers, 100 - buyers
    except Exception as e:
        return None, "error", str(e), None, 0, 50, 50

def calculate_sl_tp(price, direction, atr):
    if price is None:
        return None, None
    risk = atr * 1.1 if atr and atr > 0 else price * 0.008
    if direction == "bullish":
        return price - risk, price + (risk * 3)
    return price + risk, price - (risk * 3)

def is_relevant(title, must_contain):
    t = title.lower()
    return any(w in t for w in must_contain)

def get_news(info):
    query = " OR ".join(info["keywords"])
    params = {
        "q": query, "language": "en", "sortBy": "publishedAt",
        "pageSize": 8, "apiKey": NEWS_API_KEY,
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
        if len(title) < 15 or not is_relevant(title, info["must_contain"]):
            continue
        try:
            res = sentiment_model(title[:400])[0]
            label, score = res["label"].lower(), res["score"]
            if label in ["positive", "negative"] and score >= 0.58:
                direction = "bullish" if label == "positive" else "bearish"
                if best is None or score > best["confidence"]:
                    best = {"direction": direction, "confidence": score, "headline": title}
        except:
            continue
    return best

# Session
if "logs" not in st.session_state:
    st.session_state.logs = []
if "signals" not in st.session_state:
    st.session_state.signals = []

# Show current mode
weekend = is_weekend_now()
if weekend:
    st.info(" Weekend mode: Only analyzing markets that are still open (crypto)")
else:
    st.success(" Week mode: Full market list (Forex + Metals + Crypto)")

active_list = get_active_watchlist()
st.caption(f"Active assets right now: {', '.join([v['name'] for v in active_list.values()])}")

if st.button("🔄 Run Analysis", use_container_width=True):
    st.session_state.logs = []
    st.session_state.signals = []

    with st.spinner("Scanning active markets..."):
        for symbol, info in active_list.items():
            name = info["name"]
            st.session_state.logs.append(f"——— {name} ———")

            price, ta_dir, ta_reason, atr, ta_conf, buyers, sellers = get_price_data(symbol)
            if price is None:
                st.session_state.logs.append(f"ERROR: {ta_reason}")
                continue

            st.session_state.logs.append(f"Technical: {ta_reason}")
            st.session_state.logs.append(f"TA: {ta_dir.upper()} ({ta_conf:.0%})")
            st.session_state.logs.append(f"Buyers/Sellers: {buyers}% / {sellers}% (Technical estimate – last 12 candles)")

            news = get_news(info)
            if news:
                final_dir = news["direction"]
                conf = news["confidence"]
                source = "News + Technical"
                headline = news["headline"]
                st.session_state.logs.append(f"News: {final_dir.upper()} ({conf:.0%})")
                st.session_state.logs.append(f"Headline: {headline[:65]}...")
            else:
                final_dir = ta_dir
                conf = ta_conf
                source = "Technical only"
                headline = "No relevant news found"
                st.session_state.logs.append("News: None (strict filter)")

            if final_dir in ["bullish", "bearish"]:
                sl, tp = calculate_sl_tp(price, final_dir, atr)
                price_txt = f"${price:,.2f}"
                sl_txt = f"${sl:,.2f}"
                tp_txt = f"${tp:,.2f}"
                risk = abs(price - sl)
                reward = abs(tp - price)
                rr = reward / risk if risk > 0 else 3.0

                signal = {
                    "name": name, "direction": final_dir, "price": price_txt,
                    "sl": sl_txt, "tp": tp_txt, "confidence": conf,
                    "ta": ta_reason, "buyers": buyers, "sellers": sellers,
                    "headline": headline, "source": source, "rr": rr,
                    "time": datetime.utcnow().strftime("%H:%M UTC")
                }
                st.session_state.signals.append(signal)
                st.session_state.logs.append(f"✅ ENTRY {final_dir.upper()} | Conf {conf:.0%} | RR 1:{rr:.1f}")

                if send_telegram_alerts:
                    emoji = "🟢" if final_dir == "bullish" else "🔴"
                    msg = (
                        f"{emoji} <b>ENTRY {final_dir.upper()}</b> – {name}\n\n"
                        f"Entry: {price_txt}\nStop Loss: {sl_txt}\nTake Profit: {tp_txt}\n"
                        f"Risk:Reward = 1:{rr:.1f}\nConfidence: {conf:.0%}\n"
                        f"Buyers/Sellers: {buyers}% / {sellers}%\n"
                        f"(Technical estimate – last 12 candles)\n"
                        f"Technical: {ta_reason}\nSource: {source}\nTime: {signal['time']}\n\n{headline}"
                    )
                    ok = send_telegram(msg)
                    st.session_state.logs.append("Telegram: Sent" if ok else "Telegram: Failed")
            else:
                st.session_state.logs.append("Result: Neutral – no entry")

    st.success("Analysis complete")

# Display
st.subheader("📊 Entry Signals")
if st.session_state.signals:
    for s in st.session_state.signals:
        color = "green" if s["direction"] == "bullish" else "red"
        st.markdown(
            f":{color}[**ENTRY {s['direction'].upper()} – {s['name']}**]\n\n"
            f"**Entry:** {s['price']}\n**Stop Loss:** {s['sl']}\n**Take Profit:** {s['tp']}\n"
            f"**Risk:Reward:** 1:{s['rr']:.1f}\n**Confidence:** {s['confidence']:.0%}\n"
            f"**Buyers/Sellers:** {s['buyers']}% / {s['sellers']}%\n"
            f"*Technical estimate (last 12 candles)*\n"
            f"**Technical:** {s['ta']}\n**Source:** {s['source']}\n**Time:** {s['time']}\n{s['headline']}"
        )
else:
    st.info("No clear Entry signals right now.")

st.subheader("🔍 Log")
for line in st.session_state.logs:
    st.text(line)