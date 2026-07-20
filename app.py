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


def fetch_batch_with_retry(max_retries=3):
    """
    Fetch all watchlist tickers in a single yfinance call (much friendlier
    to rate limits than one call per ticker), retrying with backoff on
    rate-limit errors.
    """
    delay = 20  # seconds
    for attempt in range(max_retries):
        try:
            data = yf.download(
                WATCHLIST, period="5d", interval="5m",
                progress=False, group_by="ticker", threads=False,
            )
            if data is not None and not data.empty:
                return data
        except Exception as e:
            if "Rate limit" in str(e) or "Too Many Requests" in str(e):
                time.sleep(delay)
                delay *= 2
                continue
            raise
        time.sleep(delay)
        delay *= 2
    return None


def analyze_ticker(ticker, batch_df):
    """Compute signal for one ticker from an already-fetched batch DataFrame."""
    try:
        if batch_df is None:
            return {"ticker": ticker.replace(".NS", ""), "error": "Rate limited, retrying next cycle"}

        # With multiple tickers, yfinance returns a MultiIndex column DataFrame
        if isinstance(batch_df.columns, pd.MultiIndex):
            if ticker not in batch_df.columns.get_level_values(0):
                return {"ticker": ticker.replace(".NS", ""), "error": "No data returned"}
            df = batch_df[ticker].dropna(how="all")
        else:
            df = batch_df.dropna(how="all")

        if df.empty or len(df) < LONG_WINDOW + 1:
            return {"ticker": ticker.replace(".NS", ""), "error": "Not enough data"}

        close = df["Close"]
        sma_short = close.rolling(window=SHORT_WINDOW).mean()
        sma_long = close.rolling(window=LONG_WINDOW).mean()
        rsi = compute_rsi(close)

        latest_close = float(close.iloc[-1])
        latest_short, prev_short = float(sma_short.iloc[-1]), float(sma_short.iloc[-2])
        latest_long, prev_long = float(sma_long.iloc[-1]), float(sma_long.iloc[-2])
        rsi_val = float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else None

        signal = "HOLD"
        reason = []

        crossed_up = prev_short <= prev_long and latest_short > latest_long
        crossed_down = prev_short >= prev_long and latest_short < latest_long

        if crossed_up:
            signal = "BUY"
            reason.append(f"SMA{SHORT_WINDOW} crossed above SMA{LONG_WINDOW}")
        elif crossed_down:
            signal = "SELL"
            reason.append(f"SMA{SHORT_WINDOW} crossed below SMA{LONG_WINDOW}")

        if rsi_val is not None:
            if rsi_val < 30 and signal != "SELL":
                signal = "BUY"
                reason.append(f"RSI oversold ({rsi_val:.1f})")
            elif rsi_val > 70 and signal != "BUY":
                signal = "SELL"
                reason.append(f"RSI overbought ({rsi_val:.1f})")

        return {
            "ticker": ticker.replace(".NS", ""),
            "price": round(latest_close, 2),
            "sma_short": round(latest_short, 2) if pd.notna(latest_short) else None,
            "sma_long": round(latest_long, 2) if pd.notna(latest_long) else None,
            "rsi": round(rsi_val, 2) if rsi_val is not None else None,
            "signal": signal,
            "reason": "; ".join(reason) if reason else "No crossover / neutral RSI",
        }
    except Exception as e:
        return {"ticker": ticker.replace(".NS", ""), "error": str(e)}


def refresh_loop():
    """Background thread: refresh all tickers every 5 minutes (batched call)."""
    global latest_data
    while True:
        batch_df = fetch_batch_with_retry()
        results = [analyze_ticker(t, batch_df) for t in WATCHLIST]
        with data_lock:
            latest_data = {"stocks": results, "updated": pd.Timestamp.now().strftime("%H:%M:%S")}
        time.sleep(300)  # 5 minutes between refreshes to stay well under rate limits


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


# Start the background refresh thread. This runs whether the app is
# started directly (python3 app.py) or imported by gunicorn on Render.
# It does its first fetch immediately, so the dashboard fills in shortly
# after the page loads (poll /api/signals — it starts as {} for a few
# seconds until the first batch completes).
latest_data = {"stocks": [], "updated": None}
_refresh_thread = threading.Thread(target=refresh_loop, daemon=True)
_refresh_thread.start()

if __name__ == "__main__":
    app.run(debug=False, port=5000)
