"""
Day Trading Signal App - Backend
Fetches live price data for a watchlist of NSE stocks, computes simple
technical indicators (SMA crossover + RSI), and serves buy/sell signals
as JSON to a browser dashboard.
"""

from flask import Flask, jsonify, request
import yfinance as yf
import pandas as pd
import json
import os
import threading
import time

app = Flask(__name__)

# ---- Config ----
# NSE tickers need a ".NS" suffix for yfinance
WATCHLIST = ["SUZLON.NS", "IDEA.NS", "IEX.NS", "YESBANK.NS", "TATAPOWER.NS"]

SHORT_WINDOW = 9    # short SMA period (in candles)
LONG_WINDOW = 21     # long SMA period (in candles)
RSI_PERIOD = 14

FUNDS_FILE = os.path.join(os.path.dirname(__file__), "funds.json")

# In-memory cache of the latest computed signals
latest_data = {}
data_lock = threading.Lock()


def load_funds():
    if os.path.exists(FUNDS_FILE):
        with open(FUNDS_FILE, "r") as f:
            return json.load(f).get("available_funds", 0)
    return 0


def save_funds(amount):
    with open(FUNDS_FILE, "w") as f:
        json.dump({"available_funds": amount}, f)


def compute_rsi(series, period=RSI_PERIOD):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def analyze_ticker(ticker):
    """Pull recent price history and compute signal for one ticker."""
    try:
        df = yf.download(ticker, period="5d", interval="5m", progress=False)
        if df.empty or len(df) < LONG_WINDOW + 1:
            return {"ticker": ticker, "error": "Not enough data"}

        # yfinance sometimes returns multi-index columns; flatten if needed
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        close = df["Close"]

        df["SMA_short"] = close.rolling(window=SHORT_WINDOW).mean()
        df["SMA_long"] = close.rolling(window=LONG_WINDOW).mean()
        df["RSI"] = compute_rsi(close)

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        signal = "HOLD"
        reason = []

        # SMA crossover logic
        crossed_up = prev["SMA_short"] <= prev["SMA_long"] and latest["SMA_short"] > latest["SMA_long"]
        crossed_down = prev["SMA_short"] >= prev["SMA_long"] and latest["SMA_short"] < latest["SMA_long"]

        if crossed_up:
            signal = "BUY"
            reason.append(f"SMA{SHORT_WINDOW} crossed above SMA{LONG_WINDOW}")
        elif crossed_down:
            signal = "SELL"
            reason.append(f"SMA{SHORT_WINDOW} crossed below SMA{LONG_WINDOW}")

        # RSI overlay
        rsi_val = float(latest["RSI"]) if pd.notna(latest["RSI"]) else None
        if rsi_val is not None:
            if rsi_val < 30 and signal != "SELL":
                signal = "BUY"
                reason.append(f"RSI oversold ({rsi_val:.1f})")
            elif rsi_val > 70 and signal != "BUY":
                signal = "SELL"
                reason.append(f"RSI overbought ({rsi_val:.1f})")

        return {
            "ticker": ticker.replace(".NS", ""),
            "price": round(float(latest["Close"]), 2),
            "sma_short": round(float(latest["SMA_short"]), 2) if pd.notna(latest["SMA_short"]) else None,
            "sma_long": round(float(latest["SMA_long"]), 2) if pd.notna(latest["SMA_long"]) else None,
            "rsi": round(rsi_val, 2) if rsi_val is not None else None,
            "signal": signal,
            "reason": "; ".join(reason) if reason else "No crossover / neutral RSI",
            "updated": pd.Timestamp.now().strftime("%H:%M:%S"),
        }
    except Exception as e:
        return {"ticker": ticker.replace(".NS", ""), "error": str(e)}


def refresh_loop():
    """Background thread: refresh all tickers every 60 seconds."""
    global latest_data
    while True:
        results = [analyze_ticker(t) for t in WATCHLIST]
        with data_lock:
            latest_data = {"stocks": results, "updated": pd.Timestamp.now().strftime("%H:%M:%S")}
        time.sleep(60)


@app.route("/api/signals")
def get_signals():
    with data_lock:
        return jsonify(latest_data)


@app.route("/api/funds", methods=["GET", "POST"])
def funds():
    if request.method == "POST":
        body = request.get_json(force=True)
        amount = float(body.get("available_funds", 0))
        save_funds(amount)
        return jsonify({"available_funds": amount})
    return jsonify({"available_funds": load_funds()})


@app.route("/")
def index():
    return app.send_static_file("index.html")


if __name__ == "__main__":
    # Do one synchronous fetch immediately so the dashboard isn't empty on load
    latest_data = {"stocks": [analyze_ticker(t) for t in WATCHLIST], "updated": pd.Timestamp.now().strftime("%H:%M:%S")}
    t = threading.Thread(target=refresh_loop, daemon=True)
    t.start()
    app.run(debug=False, port=5000)
