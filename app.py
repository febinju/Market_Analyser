"""
Day Trading Signal App - Backend
Fetches live price data for a watchlist of NSE stocks via yfinance,
computes simple technical indicators (SMA crossover + RSI), and serves
buy/sell signals as JSON to a browser dashboard.
"""

from flask import Flask, jsonify, request
import yfinance as yf
import pandas as pd
import json
import os
import threading
import time
import traceback

app = Flask(__name__)

# ---- Config ----
# Stored without ".NS" suffix; added back only when calling yfinance
DEFAULT_WATCHLIST = ["SUZLON", "IDEA", "IEX", "YESBANK", "TATAPOWER"]
MAX_WATCHLIST_SIZE = 5

SHORT_WINDOW = 9    # short SMA period (in candles)
LONG_WINDOW = 21     # long SMA period (in candles)
RSI_PERIOD = 14

FUNDS_FILE = os.path.join(os.path.dirname(__file__), "funds.json")
WATCHLIST_FILE = os.path.join(os.path.dirname(__file__), "watchlist.json")

# In-memory cache of the latest computed signals
latest_data = {"stocks": [], "updated": None}
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


def to_yf_symbol(ticker):
    return f"{ticker}.NS"


def fetch_batch_with_retry(tickers, max_retries=4):
    """
    Fetch all watchlist tickers in a single yfinance call (much friendlier
    to rate limits than one call per ticker), retrying with backoff on
    rate-limit errors.
    """
    if not tickers:
        return None
    yf_symbols = [to_yf_symbol(t) for t in tickers]
    delay = 20  # seconds
    for attempt in range(max_retries):
        try:
            data = yf.download(
                yf_symbols, period="5d", interval="5m",
                progress=False, group_by="ticker", threads=False,
            )
            print(f"[yfinance] attempt {attempt + 1}: got data, empty={data.empty if data is not None else 'N/A'}", flush=True)
            if data is not None and not data.empty:
                return data
        except Exception as e:
            print(f"[yfinance] Exception on attempt {attempt + 1}: {e}", flush=True)
            if "Rate limit" in str(e) or "Too Many Requests" in str(e):
                time.sleep(delay)
                delay *= 2
                continue
        time.sleep(delay)
        delay *= 2
    return None


def analyze_ticker(ticker, batch_df):
    """Compute signal for one ticker from an already-fetched batch DataFrame."""
    try:
        if batch_df is None:
            return {"ticker": ticker, "error": "Rate limited, retrying next cycle"}

        yf_symbol = to_yf_symbol(ticker)

        if isinstance(batch_df.columns, pd.MultiIndex):
            if yf_symbol not in batch_df.columns.get_level_values(0):
                return {"ticker": ticker, "error": "No data returned"}
            df = batch_df[yf_symbol].dropna(how="all")
        else:
            df = batch_df.dropna(how="all")

        if df.empty or len(df) < LONG_WINDOW + 1:
            return {"ticker": ticker, "error": "Not enough data"}

        close = df["Close"]
        sma_short = close.rolling(window=SHORT_WINDOW).mean()
        sma_long = close.rolling(window=LONG_WINDOW).mean()
        rsi = compute_rsi(close)

        # Compute a signal for every candle (vectorized), not just the latest,
        # so we can chart the full intraday history of buy/sell points.
        crossed_up_series = (sma_short.shift(1) <= sma_long.shift(1)) & (sma_short > sma_long)
        crossed_down_series = (sma_short.shift(1) >= sma_long.shift(1)) & (sma_short < sma_long)
        rsi_oversold = rsi < 30
        rsi_overbought = rsi > 70

        signals = pd.Series("HOLD", index=df.index)
        signals[crossed_up_series] = "BUY"
        signals[crossed_down_series] = "SELL"
        signals[rsi_oversold & (signals != "SELL")] = "BUY"
        signals[rsi_overbought & (signals != "BUY")] = "SELL"

        # Restrict intraday history to today's session only (same calendar
        # date as the most recent candle, robust to timezone quirks)
        last_date = df.index[-1].date()
        today_mask = df.index.date == last_date
        history = [
            {
                "time": ts.strftime("%H:%M"),
                "price": round(float(price), 2),
                "signal": sig,
            }
            for ts, price, sig in zip(df.index[today_mask], close[today_mask], signals[today_mask])
        ]

        latest_close = float(close.iloc[-1])
        latest_short = float(sma_short.iloc[-1])
        latest_long = float(sma_long.iloc[-1])
        rsi_val = float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else None
        signal = signals.iloc[-1]

        reason = []
        if signal == "BUY":
            if crossed_up_series.iloc[-1]:
                reason.append(f"SMA{SHORT_WINDOW} crossed above SMA{LONG_WINDOW}")
            if rsi_val is not None and rsi_val < 30:
                reason.append(f"RSI oversold ({rsi_val:.1f})")
        elif signal == "SELL":
            if crossed_down_series.iloc[-1]:
                reason.append(f"SMA{SHORT_WINDOW} crossed below SMA{LONG_WINDOW}")
            if rsi_val is not None and rsi_val > 70:
                reason.append(f"RSI overbought ({rsi_val:.1f})")

        return {
            "ticker": ticker,
            "price": round(latest_close, 2),
            "sma_short": round(latest_short, 2) if pd.notna(latest_short) else None,
            "sma_long": round(latest_long, 2) if pd.notna(latest_long) else None,
            "rsi": round(rsi_val, 2) if rsi_val is not None else None,
            "signal": signal,
            "reason": "; ".join(reason) if reason else "No crossover / neutral RSI",
            "history": history,
        }
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


def do_refresh():
    """Fetch + analyze the current watchlist, updating latest_data."""
    global latest_data
    with watchlist_lock:
        tickers = load_watchlist()
    batch_df = fetch_batch_with_retry(tickers)
    results = [analyze_ticker(t, batch_df) for t in tickers]
    with data_lock:
        latest_data = {"stocks": results, "updated": pd.Timestamp.now().strftime("%H:%M:%S")}


def refresh_loop():
    """Background thread: refresh the watchlist every 5 minutes (batched call)."""
    while True:
        try:
            do_refresh()
        except Exception:
            print(f"[refresh_loop] Unhandled exception:\n{traceback.format_exc()}", flush=True)
        time.sleep(300)


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


# Start the background refresh thread lazily, on the first incoming
# request, rather than at import time. Some gunicorn configurations fork
# worker processes after importing the app module, and threads started
# during that import do not survive fork() — only the main thread does.
# Starting on first request guarantees this runs inside the actual
# serving process.
_thread_started = False
_thread_start_lock = threading.Lock()


def ensure_refresh_thread():
    global _thread_started
    with _thread_start_lock:
        if not _thread_started:
            _thread_started = True
            threading.Thread(target=refresh_loop, daemon=True).start()
            print("[startup] background refresh thread started", flush=True)


@app.before_request
def _start_thread_once():
    ensure_refresh_thread()


if __name__ == "__main__":
    ensure_refresh_thread()
    app.run(debug=False, port=5000)