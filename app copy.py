"""
Day Trading Signal App - Backend
Fetches live price data for a watchlist of NSE stocks via the Twelve Data
API, computes simple technical indicators (SMA crossover + RSI), and
serves buy/sell signals as JSON to a browser dashboard.
"""

from flask import Flask, jsonify, request
import pandas as pd
import requests
import json
import os
import threading
import time

app = Flask(__name__)

# ---- Config ----
DEFAULT_WATCHLIST = ["SUZLON", "IDEA", "IEX", "YESBANK", "TATAPOWER"]
MAX_WATCHLIST_SIZE = 5

SHORT_WINDOW = 9    # short SMA period (in candles)
LONG_WINDOW = 21     # long SMA period (in candles)
RSI_PERIOD = 14

# Set this as an environment variable named TWELVE_DATA_API_KEY
# (on Render: Dashboard -> your service -> Environment -> Add Environment Variable)
TWELVE_DATA_API_KEY = os.environ.get("TWELVE_DATA_API_KEY", "")
TWELVE_DATA_URL = "https://api.twelvedata.com/time_series"

FUNDS_FILE = os.path.join(os.path.dirname(__file__), "funds.json")
WATCHLIST_FILE = os.path.join(os.path.dirname(__file__), "watchlist.json")

# In-memory cache of the latest computed signals
latest_data = {}
data_lock = threading.Lock()
watchlist_lock = threading.Lock()


def load_funds():
    if os.path.exists(FUNDS_FILE):
        with open(FUNDS_FILE, "r") as f:
            return json.load(f).get("available_funds", 0)
    return 0


def save_funds(amount):
    with open(FUNDS_FILE, "w") as f:
        json.dump({"available_funds": amount}, f)


def load_watchlist():
    if os.path.exists(WATCHLIST_FILE):
        with open(WATCHLIST_FILE, "r") as f:
            return json.load(f).get("tickers", DEFAULT_WATCHLIST)
    return list(DEFAULT_WATCHLIST)


def save_watchlist(tickers):
    with open(WATCHLIST_FILE, "w") as f:
        json.dump({"tickers": tickers}, f)


def compute_rsi(series, period=RSI_PERIOD):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def to_td_symbol(ticker):
    """Twelve Data expects NSE symbols as e.g. 'RELIANCE:NSE'."""
    return f"{ticker}:NSE"


def fetch_batch_with_retry(tickers, max_retries=3):
    """
    Fetch all watchlist tickers in a single Twelve Data time_series call
    (comma-separated symbols), retrying with backoff if rate limited.
    """
    if not tickers:
        return None
    if not TWELVE_DATA_API_KEY:
        return "NO_API_KEY"

    symbols = ",".join(to_td_symbol(t) for t in tickers)
    delay = 15  # seconds
    for attempt in range(max_retries):
        try:
            resp = requests.get(
                TWELVE_DATA_URL,
                params={
                    "symbol": symbols,
                    "interval": "5min",
                    "outputsize": 50,
                    "apikey": TWELVE_DATA_API_KEY,
                },
                timeout=15,
            )
            data = resp.json()
            # Twelve Data returns {"code": 429, ...} when rate limited
            if isinstance(data, dict) and data.get("code") in (429, 8, 429):
                time.sleep(delay)
                delay *= 2
                continue
            return data
        except Exception:
            time.sleep(delay)
            delay *= 2
    return None


def analyze_ticker(ticker, batch_data):
    """Compute signal for one ticker from the already-fetched batch response."""
    try:
        if batch_data == "NO_API_KEY":
            return {"ticker": ticker, "error": "Missing TWELVE_DATA_API_KEY environment variable"}
        if not batch_data:
            return {"ticker": ticker, "error": "Rate limited or no response, retrying next cycle"}

        symbol_key = to_td_symbol(ticker)

        # When multiple symbols are requested, Twelve Data nests each under
        # its symbol key. When only one symbol is requested, the response is
        # flat (meta/values/status at the top level).
        if isinstance(batch_data, dict) and symbol_key in batch_data:
            entry = batch_data[symbol_key]
        elif isinstance(batch_data, dict) and "values" in batch_data:
            entry = batch_data
        else:
            entry = None

        if entry is None or entry.get("status") == "error":
            msg = entry.get("message") if entry else "No data returned"
            return {"ticker": ticker, "error": msg}

        values = entry.get("values", [])
        if len(values) < LONG_WINDOW + 1:
            return {"ticker": ticker, "error": "Not enough data"}

        # Twelve Data returns most-recent-first; sort ascending for rolling calcs
        values = sorted(values, key=lambda v: v["datetime"])
        close = pd.Series([float(v["close"]) for v in values])

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
            "ticker": ticker,
            "price": round(latest_close, 2),
            "sma_short": round(latest_short, 2) if pd.notna(latest_short) else None,
            "sma_long": round(latest_long, 2) if pd.notna(latest_long) else None,
            "rsi": round(rsi_val, 2) if rsi_val is not None else None,
            "signal": signal,
            "reason": "; ".join(reason) if reason else "No crossover / neutral RSI",
        }
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


def do_refresh():
    """Fetch + analyze the current watchlist, updating latest_data."""
    global latest_data
    with watchlist_lock:
        tickers = load_watchlist()
    batch_data = fetch_batch_with_retry(tickers)
    results = [analyze_ticker(t, batch_data) for t in tickers]
    with data_lock:
        latest_data = {"stocks": results, "updated": pd.Timestamp.now().strftime("%H:%M:%S")}


def refresh_loop():
    """Background thread: refresh the watchlist every 2 minutes.
    5 symbols x 30 refreshes/hour = 150 credits/hour, well under the
    Twelve Data free tier's 800/day limit."""
    while True:
        do_refresh()
        time.sleep(120)


@app.route("/api/signals")
def get_signals():
    with data_lock:
        return jsonify(latest_data)


@app.route("/api/watchlist", methods=["GET"])
def get_watchlist():
    with watchlist_lock:
        tickers = load_watchlist()
    return jsonify({"tickers": tickers})


@app.route("/api/watchlist/add", methods=["POST"])
def add_to_watchlist():
    body = request.get_json(force=True)
    ticker = (body.get("ticker") or "").strip().upper()
    if not ticker:
        return jsonify({"error": "No ticker provided"}), 400

    with watchlist_lock:
        tickers = load_watchlist()
        if ticker in tickers:
            return jsonify({"error": f"{ticker} is already in the watchlist"}), 400
        if len(tickers) >= MAX_WATCHLIST_SIZE:
            return jsonify({"error": f"Watchlist is full (max {MAX_WATCHLIST_SIZE} stocks). Remove one first."}), 400
        tickers.append(ticker)
        save_watchlist(tickers)

    # Refresh immediately in the background so the new stock shows up without
    # waiting for the next scheduled cycle
    threading.Thread(target=do_refresh, daemon=True).start()

    return jsonify({"tickers": tickers})


@app.route("/api/watchlist/remove", methods=["POST"])
def remove_from_watchlist():
    body = request.get_json(force=True)
    ticker = (body.get("ticker") or "").strip().upper()

    with watchlist_lock:
        tickers = load_watchlist()
        if ticker not in tickers:
            return jsonify({"error": f"{ticker} is not in the watchlist"}), 400
        tickers.remove(ticker)
        save_watchlist(tickers)

    threading.Thread(target=do_refresh, daemon=True).start()

    return jsonify({"tickers": tickers})


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
latest_data = {"stocks": [], "updated": None}
_refresh_thread = threading.Thread(target=refresh_loop, daemon=True)
_refresh_thread.start()

if __name__ == "__main__":
    app.run(debug=False, port=5000)
