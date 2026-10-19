import streamlit as st
import requests
import yfinance as yf
from datetime import datetime, timedelta
from transformers import pipeline

st.set_page_config(page_title="News + TA Bot", page_icon="📈", layout="wide")
st.title("📈 News + Technical Analysis Bot")
st.caption("Short-term (15m–2h) • Flexible confidence • Stronger news")

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

# More flexible confidence
MIN_CONFIDENCE = 0.58

FULL_WATCHLIST = {
    "BTC-USD": {"name": "Bitcoin", "keywords": ["Bitcoin", "BTC", "Bitcoin price"], "must_contain": ["bitcoin", "btc"], "open_weekend": True},
    "GC=F": {"name": "Gold", "keywords": ["gold", "gold price", "XAUUSD", "bullion"], "must_contain": ["gold", "xau", "bullion"], "open_weekend": False},
    "SI=F": {"name": "Silver", "keywords": ["silver", "silver price", "XAGUSD"], "must_contain": ["silver", "xag"], "open_weekend": False},
    "CL=F": {"name": "Crude Oil", "keywords": ["oil", "crude oil", "WTI", "Brent"], "must_contain": ["oil", "wti", "brent", "crude"], "open_weekend": False},
    "EURUSD=X": {"name": "EUR/USD", "keywords": ["EURUSD", "EUR/USD", "euro"], "must_contain": ["eur", "euro"], "open_weekend": False},
    "GBPUSD=X": {"name": "GBP/USD", "keywords": ["GBPUSD", "GBP/USD", "pound"], "must_contain": ["gbp", "pound", "sterling"], "open_weekend": False},
    "AUDUSD=X": {"name": "AUD/USD", "keywords": ["AUDUSD", "AUD/USD", "aussie"], "must_contain": ["aud", "aussie", "australia"], "open_weekend": False},
    "USDJPY=X": {"name": "USD/JPY", "keywords": ["USDJPY", "USD/JPY", "yen"], "must_contain": ["jpy", "yen", "japan"], "open_weekend": False},
}

def is_weekend_now():
    return datetime.utcnow().weekday() >= 5

def get_active_watchlist():
    if is_weekend_now():
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
        df = yf.download(symbol, period="5d", interval="15m", progress=False)
        if df.empty or len(df) < 30:
            df = yf.download(symbol, period="5d", interval="1h", progress=False)

        if df.empty or len(df) < 20:
            raise Exception("No data")

        df = df.dropna()
        close = df["Close"]
        high = df["High"]
        low = df["Low"]

        price = float(close.iloc[-1])
        sma20 = float(close.rolling(20).mean().iloc[-1])
        atr = float((high - low).rolling(14).mean().iloc[-1])
        change = ((price - float(close.iloc[-12])) / float(close.iloc[-12])) * 100

        if price > sma20 * 1.001 and change > 0.15:
            direction, conf = "bullish", 0.70
        elif price < sma20 * 0.999 and change < -0.15:
            direction, conf = "bearish", 0.70
        elif price > sma20:
            direction, conf = "bullish", 0.60
        elif price < sma20:
            direction, conf = "bearish", 0.60
        else:
            direction, conf = "neutral", 0.50

        reason = f"Price {price:.2f} | SMA20 {sma20:.2f} | Chg {change:.2f}%"
        return price, direction, reason, atr, conf
    except Exception as e:
        try:
            ticker = yf.Ticker(symbol)
            price = float(ticker.fast_info["lastPrice"])
            atr = price * 0.006
            return price, "neutral", f"Price {price:.2f} (limited data)", atr, 0.50
        except:
            return None, "error", str(e), None, 0

def calculate_sl_tp(price, direction, atr):
    if price is None:
        return None, None
    risk = atr * 1.0 if atr and atr > 0 else price * 0.005
    if direction == "bullish":
        sl = price - risk
        tp = price + (risk * 3)
    else:
        sl = price + risk
        tp = price - (risk * 3)
    return sl, tp

def get_expected_hold_time(confidence, change_abs):
    if confidence >= 0.70 and change_abs > 0.30:
        return "15–30 minutes"
    elif confidence >= 0.62:
        return "30–60 minutes"
    else:
        return "1–2 hours"

def is_relevant(title, must_contain):
    t = title.lower()
    return any(w in t for w in must_contain)

def get_news(info):
    """More flexible news detection"""
    query = " OR ".join(info["keywords"])
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 12,
        "apiKey": NEWS_API_KEY,
        "from": (datetime.utcnow() - timedelta(hours=24)).isoformat()
    }
    try:
        r = requests.get("https://newsapi.org/v2/everything", params=params, timeout=12)
        data = r.json()
        articles = data.get("articles", [])
    except Exception as e:
        return None

    best = None
    for art in articles:
        title = art.get("title") or ""
        if len(title) < 12:
            continue

        # Soft relevance check
        if not is_relevant(title, info["must_contain"]):
            continue

        try:
            res = sentiment_model(title[:450])[0]
            label = res["label"].lower()
            score = res["score"]
            if label in ["positive", "negative"] and score >= MIN_CONFIDENCE:
                direction = "bullish" if label == "positive" else "bearish"
                if best is None or score > best["confidence"]:
                    best = {
                        "direction": direction,
                        "confidence": score,
                        "headline": title
                    }
        except:
            continue
    return best

# Session
if "logs" not in st.session_state:
    st.session_state.logs = []
if "signals" not in st.session_state:
    st.session_state.signals = []

weekend = is_weekend_now()
if weekend:
    st.info("Weekend mode: Only open markets (crypto)")
else:
    st.success("Week mode: Full list active")

active_list = get_active_watchlist()
st.caption(f"Active: {', '.join([v['name'] for v in active_list.values()])} | Min Confidence: {int(MIN_CONFIDENCE*100)}%")

if st.button("🔄 Run Analysis", use_container_width=True):
    st.session_state.logs = []
    st.session_state.signals = []

    with st.spinner("Scanning news + technicals..."):
        for symbol, info in active_list.items():
            name = info["name"]
            st.session_state.logs.append(f"——— {name} ———")

            price, ta_dir, ta_reason, atr, ta_conf = get_price_data(symbol)
            if price is None:
                st.session_state.logs.append(f"ERROR: {ta_reason}")
                continue

            st.session_state.logs.append(f"Technical: {ta_reason}")
            st.session_state.logs.append(f"TA: {ta_dir.upper()} ({ta_conf*100:.0f}%)")

            news = get_news(info)

            if news:
                final_dir = news["direction"]
                conf = news["confidence"]
                source = "News + Technical"
                headline = news["headline"]
                st.session_state.logs.append(f"News: {final_dir.upper()} ({conf*100:.0f}%)")
                st.session_state.logs.append(f"Headline: {headline[:70]}...")
            else:
                final_dir = ta_dir
                conf = ta_conf
                source = "Technical only"
                headline = "No relevant news found"
                st.session_state.logs.append("News: None")

            if final_dir in ["bullish", "bearish"] and conf >= MIN_CONFIDENCE:
                sl, tp = calculate_sl_tp(price, final_dir, atr)
                price_txt = f"${price:,.2f}"
                sl_txt = f"${sl:,.2f}"
                tp_txt = f"${tp:,.2f}"
                risk = abs(price - sl)
                reward = abs(tp - price)
                rr = reward / risk if risk > 0 else 3.0

                change_abs = 0.25
                if "Chg" in ta_reason:
                    try:
                        change_abs = abs(float(ta_reason.split("Chg ")[-1].replace("%", "")))
                    except:
                        pass
                hold_time = get_expected_hold_time(conf, change_abs)

                signal = {
                    "name": name,
                    "direction": final_dir,
                    "price": price_txt,
                    "sl": sl_txt,
                    "tp": tp_txt,
                    "confidence": conf,
                    "ta": ta_reason,
                    "headline": headline,
                    "source": source,
                    "rr": rr,
                    "hold": hold_time,
                    "time": datetime.utcnow().strftime("%H:%M UTC")
                }
                st.session_state.signals.append(signal)
                st.session_state.logs.append(
                    f"✅ ENTRY {final_dir.upper()} | Confidence {conf*100:.0f}% | Hold: {hold_time}"
                )

                if send_telegram_alerts:
                    emoji = "🟢" if final_dir == "bullish" else "🔴"
                    msg = (
                        f"{emoji} <b>ENTRY {final_dir.upper()}</b> – {name}\n\n"
                        f"Entry: {price_txt}\n"
                        f"Stop Loss: {sl_txt}\n"
                        f"Take Profit: {tp_txt}\n"
                        f"Risk:Reward = 1:{rr:.1f}\n"
                        f"Confidence: {conf*100:.0f}%\n"
                        f"Expected Hold: {hold_time}\n"
                        f"Technical: {ta_reason}\n"
                        f"Source: {source}\n"
                        f"Time: {signal['time']}\n\n"
                        f"{headline}"
                    )
                    ok = send_telegram(msg)
                    st.session_state.logs.append("Telegram: Sent" if ok else "Telegram: Failed")
            else:
                st.session_state.logs.append(
                    f"Skipped (Dir={final_dir}, Conf={conf*100:.0f}%)"
                )

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
            f"**Confidence:** {s['confidence']*100:.0f}%\n"
            f"**Expected Hold:** {s['hold']}\n"
            f"**Technical:** {s['ta']}\n"
            f"**Source:** {s['source']}\n"
            f"**Time:** {s['time']}\n"
            f"{s['headline']}"
        )
else:
    st.info("No signals meeting the current criteria.")

st.subheader("🔍 Log")
for line in st.session_state.logs:
    st.text(line)