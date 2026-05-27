from __future__ import annotations

import math
import os
import json
import re
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from flask import Flask, jsonify, render_template_string, request


class SimpleMemoryCache:
    def __init__(self):
        self._cache = {}
        self._lock = threading.Lock()

    def get(self, key, ttl_seconds):
        with self._lock:
            if key in self._cache:
                val, expiry = self._cache[key]
                if time.time() < expiry:
                    return val
                else:
                    del self._cache[key]
            return None

    def set(self, key, value, ttl_seconds):
        with self._lock:
            self._cache[key] = (value, time.time() + ttl_seconds)

    def clear(self):
        with self._lock:
            self._cache.clear()


_global_mem_cache = SimpleMemoryCache()
_yf_lock = threading.Lock()


def clear_backend_cache():
    _global_mem_cache.clear()


ROOT = Path(__file__).resolve().parent
PORT = 8766
ENV_FILE = ROOT / "alpaca_credentials.env"
ALPACA_CLI_CACHE = ROOT / "alpaca_live_cache.json"
ALPACA_EXE = Path(os.getenv("ALPACA_CLI_PATH", "alpaca"))
ALPACA_PROFILE = "codex-dashboard"
MARKOV_PROMPT_FILE = ROOT / "codex_markov_hedge_fund_prompt.md"

app = Flask(__name__)


def load_local_env() -> None:
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"'))


load_local_env()


POSITIVE_WORDS = {
    "beat", "beats", "raise", "raised", "growth", "upgrade", "upgrades", "bullish", "record",
    "strong", "surge", "surges", "rally", "rises", "profit", "profits", "approval", "deal",
    "wins", "expands", "outperform", "optimism", "accelerates", "launch", "breakthrough",
}
NEGATIVE_WORDS = {
    "miss", "misses", "cut", "cuts", "downgrade", "downgrades", "bearish", "probe", "lawsuit",
    "fall", "falls", "drops", "slump", "weak", "warning", "warns", "loss", "losses",
    "layoff", "tariff", "recall", "delay", "risk", "concern", "selloff", "investigation",
}
CATALYST_KEYWORDS = {
    "AI / compute": ["ai", "artificial intelligence", "chip", "gpu", "data center", "cloud", "inference"],
    "Fed / rates": ["fed", "federal reserve", "rate", "rates", "yield", "treasury", "inflation", "cpi", "pce"],
    "Government / regulation": ["government", "regulation", "regulator", "antitrust", "doj", "ftc", "sec", "ban", "approval"],
    "Tariff / geopolitics": ["tariff", "sanction", "china", "taiwan", "iran", "war", "export control", "trade"],
    "IPO / issuance": ["ipo", "offering", "secondary", "share sale", "insider sale", "stake sale", "dilution"],
    "Earnings / guidance": ["earnings", "revenue", "profit", "margin", "guidance", "forecast", "outlook"],
    "M&A / deal": ["acquire", "acquisition", "merger", "deal", "partnership", "stake"],
}


def safe_float(value, default=None):
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def pct(value, default=None):
    value = safe_float(value)
    if value is None:
        return default
    return round(value * 100, 2)


def clean_ticker(ticker: str) -> str:
    ticker = ticker.strip().upper()
    return re.sub(r"[^A-Z0-9.\-=^]", "", ticker)[:16] or "NVDA"


@lru_cache(maxsize=256)
def download(ticker: str, period: str = "3y") -> pd.DataFrame:
    cache_key = f"download:{ticker}:{period}"
    cached = _global_mem_cache.get(cache_key, 600)
    if cached is not None:
        return cached.copy()
    with _yf_lock:
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False, threads=False)
    if df.empty:
        raise ValueError(f"No price data found for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(how="all")
    df.index = pd.to_datetime(df.index).tz_localize(None)
    _global_mem_cache.set(cache_key, df, 600)
    return df.copy()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["Close"]
    out["SMA20"] = close.rolling(20).mean()
    out["SMA50"] = close.rolling(50).mean()
    out["SMA200"] = close.rolling(200).mean()
    out["EMA12"] = close.ewm(span=12, adjust=False).mean()
    out["EMA26"] = close.ewm(span=26, adjust=False).mean()
    out["MACD"] = out["EMA12"] - out["EMA26"]
    out["MACD_SIGNAL"] = out["MACD"].ewm(span=9, adjust=False).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["RSI14"] = 100 - (100 / (1 + rs))
    out["BB_MID"] = out["SMA20"]
    out["BB_UPPER"] = out["BB_MID"] + 2 * close.rolling(20).std()
    out["BB_LOWER"] = out["BB_MID"] - 2 * close.rolling(20).std()
    out["VOL20"] = out["Volume"].rolling(20).mean()
    out["RET_1M"] = close.pct_change(21, fill_method=None)
    out["RET_3M"] = close.pct_change(63, fill_method=None)
    out["RET_6M"] = close.pct_change(126, fill_method=None)
    out["RET_1Y"] = close.pct_change(252, fill_method=None)
    out["DD_1Y"] = close / close.rolling(252).max() - 1
    return out


MARKOV_STATES = ["Bear", "Sideways", "Bull"]
MARKET_SHOCK_EVENTS = [
    {"date": "2018-02-05", "name": "Volmageddon volatility shock", "type": "Volatility"},
    {"date": "2018-12-24", "name": "Fed tightening / trade-war selloff", "type": "Policy"},
    {"date": "2020-03-16", "name": "COVID crash", "type": "Pandemic"},
    {"date": "2020-03-23", "name": "Fed emergency support / COVID bottom", "type": "Policy"},
    {"date": "2022-02-24", "name": "Russia invades Ukraine", "type": "War"},
    {"date": "2022-06-13", "name": "Inflation and Fed-hike shock", "type": "Rates"},
    {"date": "2023-03-10", "name": "US regional banking crisis", "type": "Credit"},
    {"date": "2023-10-27", "name": "Rates peak / bond stress", "type": "Rates"},
    {"date": "2024-08-05", "name": "Global carry-trade unwind scare", "type": "Liquidity"},
    {"date": "2025-04-07", "name": "Tariff / growth scare window", "type": "Policy"},
]


def markov_label_regimes(close: pd.Series, window: int = 20, threshold: float = 0.02) -> pd.Series:
    rolling_return = close.pct_change(window)
    labels = pd.Series(1, index=close.index, dtype=int)
    labels[rolling_return > threshold] = 2
    labels[rolling_return < -threshold] = 0
    return labels.dropna()


def markov_transition_matrix(labels: pd.Series) -> np.ndarray:
    counts = np.zeros((3, 3), dtype=float)
    arr = labels.to_numpy(dtype=int)
    for i in range(len(arr) - 1):
        counts[arr[i], arr[i + 1]] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return counts / row_sums


def markov_stationary_distribution(P: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(eigvals - 1.0))
    vec = np.real(eigvecs[:, idx])
    vec = np.abs(vec)
    return vec / vec.sum()


def markov_walk_forward(close: pd.Series, labels: pd.Series, min_train: int = 252) -> dict:
    daily_returns = close.pct_change().dropna()
    common = labels.index.intersection(daily_returns.index)
    labels = labels.loc[common]
    daily_returns = daily_returns.loc[common]
    if len(labels) < min_train + 30:
        return {"sharpe": None, "maxDrawdown": None, "trades": 0, "equity": [], "dates": []}

    strategy_returns = []
    dates = []
    for t in range(min_train, len(labels) - 1):
        P_t = markov_transition_matrix(labels.iloc[:t])
        current_state = int(labels.iloc[t])
        signal = float(P_t[current_state, 2] - P_t[current_state, 0])
        position = float(np.sign(signal))
        next_day_return = float(daily_returns.iloc[t + 1])
        strategy_returns.append(position * next_day_return)
        dates.append(daily_returns.index[t + 1].strftime("%Y-%m-%d"))

    sr = np.array(strategy_returns, dtype=float)
    sharpe = None
    if len(sr) and sr.std(ddof=1) > 0 and np.isfinite(sr.std(ddof=1)):
        sharpe = float(sr.mean() / sr.std(ddof=1) * np.sqrt(252))
    equity = (1.0 + sr).cumprod()
    if len(equity):
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max
        max_dd = float(drawdown.min())
    else:
        max_dd = None
    return {
        "sharpe": sharpe,
        "maxDrawdown": max_dd,
        "trades": int(len(sr)),
        "equity": np.round(equity, 4).tolist(),
        "dates": dates,
        "returns": np.round(sr, 8).tolist(),
    }


def markov_fast_strategy_proxy(close: pd.Series, labels: pd.Series, P: np.ndarray) -> dict:
    daily_returns = close.pct_change(fill_method=None).dropna()
    common = labels.index.intersection(daily_returns.index)
    labels = labels.loc[common]
    daily_returns = daily_returns.loc[common]
    if len(labels) < 30:
        return {"sharpe": None, "maxDrawdown": None, "trades": 0, "equity": [], "dates": [], "returns": []}

    state_ids = labels.to_numpy(dtype=int)
    signals = P[state_ids, 2] - P[state_ids, 0]
    positions = np.sign(signals[:-1])
    next_returns = daily_returns.iloc[1:].to_numpy(dtype=float)
    strategy_returns = positions * next_returns
    dates = daily_returns.index[1:].strftime("%Y-%m-%d").tolist()

    sharpe = None
    if len(strategy_returns) and strategy_returns.std(ddof=1) > 0:
        sharpe = float(strategy_returns.mean() / strategy_returns.std(ddof=1) * np.sqrt(252))
    equity = (1.0 + strategy_returns).cumprod()
    max_dd = None
    if len(equity):
        running_max = np.maximum.accumulate(equity)
        max_dd = float(((equity - running_max) / running_max).min())
    return {
        "sharpe": sharpe,
        "maxDrawdown": max_dd,
        "trades": int(len(strategy_returns)),
        "equity": np.round(equity, 4).tolist(),
        "dates": dates,
        "returns": np.round(strategy_returns, 8).tolist(),
        "mode": "fast_proxy",
    }


def event_windows_for_index(index: pd.DatetimeIndex, days: int = 20) -> list[dict]:
    idx_start = index.min()
    idx_end = index.max()
    events = []
    for event in MARKET_SHOCK_EVENTS:
        dt = pd.Timestamp(event["date"])
        if idx_start - pd.Timedelta(days=days) <= dt <= idx_end + pd.Timedelta(days=days):
            item = dict(event)
            item["windowStart"] = (dt - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
            item["windowEnd"] = (dt + pd.Timedelta(days=days)).strftime("%Y-%m-%d")
            events.append(item)
    return events


def event_adjusted_walkforward(walk: dict, events: list[dict], window_days: int = 20) -> dict:
    dates = pd.to_datetime(walk.get("dates", []))
    returns = np.array(walk.get("returns", []), dtype=float)
    if len(dates) == 0 or len(returns) == 0:
        return {"shockTrades": 0, "cleanTrades": 0, "cleanSharpe": None, "note": "No walk-forward returns available."}
    mask = np.zeros(len(dates), dtype=bool)
    for event in events:
        dt = pd.Timestamp(event["date"])
        mask |= (dates >= dt - pd.Timedelta(days=window_days)) & (dates <= dt + pd.Timedelta(days=window_days))
    clean = returns[~mask]
    clean_sharpe = None
    if len(clean) > 30 and clean.std(ddof=1) > 0:
        clean_sharpe = float(clean.mean() / clean.std(ddof=1) * np.sqrt(252))
    return {
        "shockTrades": int(mask.sum()),
        "cleanTrades": int((~mask).sum()),
        "cleanSharpe": clean_sharpe,
        "note": "Clean Sharpe excludes +/-20 calendar days around listed shock events to reduce event-overfit risk.",
    }


def markov_fit_hmm(close: pd.Series) -> dict:
    try:
        from hmmlearn import hmm
    except Exception:
        return {"available": False, "message": "HMM extension skipped. hmmlearn is not installed or failed to import."}
    try:
        returns = close.pct_change().dropna()
        X = returns.to_numpy().reshape(-1, 1)
        model = hmm.GaussianHMM(n_components=3, covariance_type="diag", n_iter=200, random_state=42)
        model.fit(X)
        means = np.array([model.means_[k][0] for k in range(model.n_components)])
        order = np.argsort(means)
        names = ["Bear (lowest mean return)", "Sideways", "Bull (highest mean return)"]
        rows = []
        for rank, state_idx in enumerate(order):
            rows.append({
                "label": names[rank],
                "state": int(state_idx),
                "meanDailyReturn": float(means[state_idx]),
            })
        return {"available": True, "message": "HMM fitted with Baum-Welch and decoded with Viterbi.", "states": rows}
    except Exception as exc:
        return {"available": False, "message": f"HMM extension skipped at runtime: {exc}"}


def run_markov_model(ticker: str, years: int = 10, window: int = 20, threshold: float = 0.02, include_hmm: bool = True) -> dict:
    cache_key = f"markov_model:{ticker}:{years}:{window}:{threshold}:{include_hmm}"
    cached = _global_mem_cache.get(cache_key, 300)
    if cached is not None:
        return cached
    res = _run_markov_model_uncached(ticker, years, window, threshold, include_hmm)
    _global_mem_cache.set(cache_key, res, 300)
    return res


def _run_markov_model_uncached(ticker: str, years: int = 10, window: int = 20, threshold: float = 0.02, include_hmm: bool = True) -> dict:
    history_period = f"{max(1, int(years))}y"
    df = download(ticker, history_period)
    close = df["Close"].dropna()
    return run_markov_from_close(ticker, close, years, window, threshold, include_hmm)


def run_markov_from_close(
    ticker: str,
    close: pd.Series,
    years: int,
    window: int,
    threshold: float,
    include_hmm: bool,
    fast: bool = False,
) -> dict:
    labels = markov_label_regimes(close, window=window, threshold=threshold)
    P = markov_transition_matrix(labels)
    stationary = markov_stationary_distribution(P)
    forecast_3 = np.linalg.matrix_power(P, 3)
    forecast_5 = np.linalg.matrix_power(P, 5)
    forecast_20 = np.linalg.matrix_power(P, 20)
    events = event_windows_for_index(close.index)
    if fast:
        walk = markov_fast_strategy_proxy(close, labels, P)
        event_adjusted = event_adjusted_walkforward(walk, events)
        event_adjusted["note"] = "Fast mode uses the current transition matrix as a historical strategy proxy for live dashboard speed."
    else:
        walk = markov_walk_forward(close, labels)
        event_adjusted = event_adjusted_walkforward(walk, events)
    current_state_id = int(labels.iloc[-1])
    current_state = MARKOV_STATES[current_state_id]
    next_probs = P[current_state_id]
    label_counts = labels.value_counts(normalize=True).to_dict()
    regime_series = pd.DataFrame({
        "date": labels.index.strftime("%Y-%m-%d"),
        "state": [MARKOV_STATES[int(x)] for x in labels.to_numpy()],
        "stateId": labels.to_numpy(dtype=int),
        "close": close.reindex(labels.index).round(2).to_numpy(),
    })
    hmm_result = markov_fit_hmm(close) if include_hmm else {"available": False, "message": "HMM skipped by request."}
    return {
        "ticker": ticker,
        "years": years,
        "window": window,
        "threshold": threshold,
        "rows": int(len(close)),
        "start": close.index.min().strftime("%Y-%m-%d"),
        "end": close.index.max().strftime("%Y-%m-%d"),
        "currentState": current_state,
        "currentStateId": current_state_id,
        "matrix": np.round(P, 4).tolist(),
        "stationary": [{"state": MARKOV_STATES[i], "probability": round(float(stationary[i]) * 100, 2)} for i in range(3)],
        "nextDay": [{"state": MARKOV_STATES[i], "probability": round(float(next_probs[i]) * 100, 2)} for i in range(3)],
        "forecast3": np.round(forecast_3, 4).tolist(),
        "forecast5": np.round(forecast_5, 4).tolist(),
        "forecast20": np.round(forecast_20, 4).tolist(),
        "persistence": [{"state": MARKOV_STATES[i], "probability": round(float(P[i, i]) * 100, 2)} for i in range(3)],
        "sampleMix": [{"state": MARKOV_STATES[i], "probability": round(float(label_counts.get(i, 0)) * 100, 2)} for i in range(3)],
        "walkForward": walk,
        "eventAdjusted": event_adjusted,
        "events": events,
        "hmm": hmm_result,
        "regimeSeries": regime_series.to_dict(orient="records"),
        "interpretation": [
            "High diagonal values mean regimes tend to persist.",
            "Stationary distribution is the long-run baseline mix implied by the matrix.",
            "Walk-forward results re-estimate the matrix at every step, so they avoid lookahead.",
            "This is historical measurement, not a prediction guarantee.",
        ],
    }


def fusion_verdict(transcript: dict, markov: dict) -> dict:
    phase = transcript["phaseModel"]["current"]
    phase_top = transcript["phaseProbabilities"][0]["probability"]
    markov_state = markov["currentState"]
    next_bull = next((x["probability"] for x in markov["nextDay"] if x["state"] == "Bull"), 0)
    next_bear = next((x["probability"] for x in markov["nextDay"] if x["state"] == "Bear"), 0)
    sharpe = markov["walkForward"].get("sharpe")
    max_dd = markov["walkForward"].get("maxDrawdown")
    clean_sharpe = markov["eventAdjusted"].get("cleanSharpe")
    vix = transcript["market"]["vix"]
    event_penalty = 0
    if markov["eventAdjusted"].get("shockTrades", 0) > 0 and sharpe is not None and clean_sharpe is not None:
        if clean_sharpe < sharpe - 0.25:
            event_penalty = 8

    score = 50
    if "Bull Stage 2" in phase:
        score += 18
    elif "Bull Stage 1" in phase:
        score += 8
    elif "Bull Stage 3" in phase:
        score -= 8
    elif "Bear Stage 1" in phase:
        score -= 10
    elif "Bear Stage 2" in phase:
        score -= 18
    elif "Bear Stage 3" in phase:
        score -= 6

    if markov_state == "Bull":
        score += 14
    elif markov_state == "Bear":
        score -= 14
    score += (next_bull - next_bear) * 0.18
    if sharpe is not None:
        score += min(10, max(-10, sharpe * 6))
    if max_dd is not None and max_dd < -0.30:
        score -= 8
    if vix > 25:
        score -= 10
    elif vix < 18:
        score += 4
    score -= event_penalty

    # Live Sentiment Adjustment
    sentiment = transcript.get("sentiment", {})
    sentiment_score = sentiment.get("score", 0)
    sentiment_label = sentiment.get("label", "Neutral")
    
    sentiment_adj = 0
    if sentiment_label == "Positive":
        sentiment_adj = min(8, sentiment_score * 0.5)
    elif sentiment_label == "Negative":
        sentiment_adj = max(-12, sentiment_score * 0.7)
        
    score += sentiment_adj
    
    # Live Intraday Change Adjustment
    live_price = transcript.get("price")
    last_close = transcript.get("lastClose")
    intraday_adj = 0
    intraday_change = 0.0
    if live_price and last_close:
        intraday_change = (live_price / last_close - 1)
        if intraday_change > 0.015:
            intraday_adj = min(6, intraday_change * 150)
        elif intraday_change < -0.015:
            intraday_adj = max(-10, intraday_change * 200)
            
    score += intraday_adj
    score = max(0, min(100, score))

    if score >= 72:
        verdict = "Risk-on / constructive"
        action = "Favorable setup. Consider long-biased research, but still validate fundamentals, earnings, and position sizing."
    elif score >= 58:
        verdict = "Constructive but selective"
        action = "Proceed selectively. Prefer leaders with clean trend and avoid chasing stretched moves."
    elif score >= 42:
        verdict = "Mixed / wait for confirmation"
        action = "No clean edge. Wait for phase agreement, better momentum, or a clearer Markov transition edge."
    elif score >= 28:
        verdict = "Defensive / bear risk"
        action = "Reduce risk. Watch for trend breaks, failed rallies, and worsening macro or sentiment."
    else:
        verdict = "High-risk / avoid"
        action = "Avoid new risk unless this is a deliberate contrarian setup with strict risk controls."

    evidence = [
        f"Transcript cycle phase: {phase} ({phase_top:.1f}% top phase probability).",
        f"Markov observable state: {markov_state}; next-day Bull {next_bull:.1f}% vs Bear {next_bear:.1f}%.",
        f"Walk-forward Sharpe: {sharpe:.3f}" if sharpe is not None else "Walk-forward Sharpe unavailable.",
        f"Max drawdown: {max_dd * 100:.2f}%" if max_dd is not None else "Max drawdown unavailable.",
        f"Event-adjusted Sharpe: {clean_sharpe:.3f}" if clean_sharpe is not None else "Event-adjusted Sharpe unavailable.",
        f"Live news & social sentiment: {sentiment_label} (combined score {sentiment_score:+.1f}, adjustment: {sentiment_adj:+.1f} points).",
    ]
    if intraday_change != 0.0:
        evidence.append(f"Live intraday change: {intraday_change*100:+.2f}% (adjustment: {intraday_adj:+.1f} points).")
    if event_penalty:
        evidence.append("Shock-window robustness penalty applied because performance weakened after excluding major event windows.")
    return {"score": round(score, 1), "verdict": verdict, "action": action, "evidence": evidence}


def build_fusion_payload(
    ticker: str,
    years: int = 10,
    window: int = 20,
    threshold: float = 0.02,
    include_hmm: bool = False,
    fast: bool = True,
) -> dict:
    cache_key = f"fusion_payload:{ticker}:{years}:{window}:{threshold}:{include_hmm}:{fast}"
    cached = _global_mem_cache.get(cache_key, 20 if fast else 300)
    if cached is not None:
        return cached
    res = _build_fusion_payload_uncached(ticker, years, window, threshold, include_hmm, fast)
    _global_mem_cache.set(cache_key, res, 20 if fast else 300)
    return res


def _build_fusion_payload_uncached(
    ticker: str,
    years: int = 10,
    window: int = 20,
    threshold: float = 0.02,
    include_hmm: bool = False,
    fast: bool = True,
) -> dict:
    history_period = f"{max(1, int(years))}y"
    price = add_indicators(download(ticker, history_period))
    spy = price if ticker == "SPY" else add_indicators(download("SPY", history_period))
    market = market_context()
    latest = price.iloc[-1]
    prev = price.iloc[-2]
    rel_6m = safe_float(latest.get("RET_6M")) - safe_float(spy.iloc[-1].get("RET_6M"), 0)
    sentiment = get_fast_sentiment(ticker) if fast else get_consolidated_sentiment(ticker)
    catalysts = catalyst_tags(sentiment["items"])
    live_quote = alpaca_latest_quote(ticker)
    display_price = live_quote.get("last") or live_quote.get("mid") or safe_float(latest["Close"])
    verdict, score, checks = indicator_verdict(latest, pd.Series(market), rel_6m)
    phase_model = cycle_phase_model(score, sentiment["news_score"], market, latest, checks, catalysts)
    transcript = {
        "ticker": ticker,
        "price": round(safe_float(display_price), 2),
        "lastClose": round(safe_float(latest["Close"]), 2),
        "dailyChange": pct(latest["Close"] / prev["Close"] - 1),
        "verdict": verdict,
        "score": round(score, 1),
        "phaseModel": phase_model,
        "phaseProbabilities": phase_model["probabilities"],
        "checks": checks,
        "sentiment": sentiment,
        "catalysts": catalysts,
        "market": market,
        "chart": chart_payload(price, ticker),
        "liveQuote": live_quote,
    }
    markov = run_markov_from_close(
        ticker,
        price["Close"].dropna(),
        years=years,
        window=window,
        threshold=threshold,
        include_hmm=include_hmm,
        fast=fast,
    )
    final = fusion_verdict(transcript, markov)
    return {
        "ticker": ticker,
        "updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "transcript": transcript,
        "markov": markov,
        "final": final,
    }


def get_news_sentiment(ticker: str):
    cache_key = f"news_sentiment:{ticker}"
    cached = _global_mem_cache.get(cache_key, 1800)
    if cached is not None:
        return cached
    try:
        with _yf_lock:
            news = yf.Ticker(ticker).news or []
    except Exception:
        news = []
    rows = []
    total = 0
    for item in news[:12]:
        title = item.get("title") or item.get("content", {}).get("title") or ""
        publisher = item.get("publisher") or item.get("content", {}).get("provider", {}).get("displayName") or ""
        link = item.get("link") or item.get("content", {}).get("canonicalUrl", {}).get("url") or ""
        ts = item.get("providerPublishTime") or item.get("content", {}).get("pubDate")
        text = title.lower()
        pos = sum(1 for w in POSITIVE_WORDS if w in text)
        neg = sum(1 for w in NEGATIVE_WORDS if w in text)
        score = pos - neg
        total += score
        rows.append({"title": title, "publisher": publisher, "link": link, "score": score, "published": str(ts or "")})
    label = "Neutral"
    if total >= 2:
        label = "Positive"
    elif total <= -2:
        label = "Negative"
    result = {"label": label, "score": total, "items": rows}
    _global_mem_cache.set(cache_key, result, 1800)
    return result


def get_reddit_sentiment(ticker: str) -> dict:
    cache_key = f"reddit_sentiment:{ticker}"
    cached = _global_mem_cache.get(cache_key, 1800)
    if cached is not None:
        return cached
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    url = f"https://www.reddit.com/r/stocks+wallstreetbets+investing/search.json?q={ticker}&restrict_sr=1&sort=new&limit=15"
    try:
        r = requests.get(url, headers=headers, timeout=8)
        if r.status_code != 200:
            url = f"https://www.reddit.com/search.json?q={ticker}&sort=new&limit=15"
            r = requests.get(url, headers=headers, timeout=8)
        if r.status_code == 200:
            posts = r.json().get("data", {}).get("children", [])
            rows = []
            total = 0
            for post in posts:
                pdata = post.get("data", {})
                title = pdata.get("title") or ""
                selftext = pdata.get("selftext") or ""
                permalink = pdata.get("permalink") or ""
                subreddit = pdata.get("subreddit") or ""
                created_utc = pdata.get("created_utc")
                full_text = (title + " " + selftext).lower()
                pos = sum(1 for w in POSITIVE_WORDS if w in full_text)
                neg = sum(1 for w in NEGATIVE_WORDS if w in full_text)
                score = pos - neg
                total += score
                rows.append({
                    "title": title,
                    "publisher": f"r/{subreddit}",
                    "link": f"https://reddit.com{permalink}",
                    "score": score,
                    "published": datetime.fromtimestamp(created_utc, timezone.utc).isoformat() if created_utc else ""
                })
            label = "Neutral"
            if total >= 2:
                label = "Positive"
            elif total <= -2:
                label = "Negative"
            result = {"label": label, "score": total, "items": rows}
            _global_mem_cache.set(cache_key, result, 1800)
            return result
    except Exception:
        pass
    fallback_res = {"label": "Neutral", "score": 0, "items": []}
    _global_mem_cache.set(cache_key, fallback_res, 300)  # Cache failure for 5 mins to avoid spamming
    return fallback_res


def get_consolidated_sentiment(ticker: str) -> dict:
    cache_key = f"consolidated_sentiment:{ticker}"
    cached = _global_mem_cache.get(cache_key, 1800)
    if cached is not None:
        return cached
    news = get_news_sentiment(ticker)
    reddit = get_reddit_sentiment(ticker)
    
    combined_score = news["score"] + reddit["score"]
    combined_items = []
    for item in news["items"]:
        combined_items.append({
            "title": item["title"],
            "publisher": item.get("publisher") or "Yahoo Finance",
            "link": item["link"],
            "score": item["score"],
            "published": item["published"],
            "type": "news"
        })
    for item in reddit["items"]:
        combined_items.append({
            "title": item["title"],
            "publisher": item["publisher"],
            "link": item["link"],
            "score": item["score"],
            "published": item["published"],
            "type": "reddit"
        })
    
    label = "Neutral"
    if combined_score >= 3:
        label = "Positive"
    elif combined_score <= -3:
        label = "Negative"
        
    result = {
        "label": label,
        "score": combined_score,
        "news_score": news["score"],
        "reddit_score": reddit["score"],
        "items": combined_items
    }
    _global_mem_cache.set(cache_key, result, 1800)
    return result


def get_fast_sentiment(ticker: str) -> dict:
    cache_key = f"fast_sentiment:{ticker}"
    cached = _global_mem_cache.get(cache_key, 300)
    if cached is not None:
        return cached

    news = get_news_sentiment(ticker)
    reddit = _global_mem_cache.get(f"reddit_sentiment:{ticker}", 1800) or {"label": "Neutral", "score": 0, "items": []}
    combined_score = news["score"] + reddit["score"]
    combined_items = []
    for item in news["items"]:
        combined_items.append({
            "title": item["title"],
            "publisher": item.get("publisher") or "Yahoo Finance",
            "link": item["link"],
            "score": item["score"],
            "published": item["published"],
            "type": "news",
        })
    for item in reddit.get("items", [])[:5]:
        combined_items.append({
            "title": item["title"],
            "publisher": item["publisher"],
            "link": item["link"],
            "score": item["score"],
            "published": item["published"],
            "type": "reddit_cached",
        })

    label = "Neutral"
    if combined_score >= 3:
        label = "Positive"
    elif combined_score <= -3:
        label = "Negative"
    result = {
        "label": label,
        "score": combined_score,
        "news_score": news["score"],
        "reddit_score": reddit["score"],
        "items": combined_items,
    }
    _global_mem_cache.set(cache_key, result, 300)
    return result


def cached_or_neutral_sentiment(ticker: str) -> dict:
    cached = _global_mem_cache.get(f"consolidated_sentiment:{ticker}", 1800)
    if cached is not None:
        return cached
    return {"label": "Neutral", "score": 0, "news_score": 0, "reddit_score": 0, "items": []}


def catalyst_tags(news_items: list[dict]) -> list[dict]:
    text = " ".join((item.get("title") or "").lower() for item in news_items)
    tags = []
    for tag, words in CATALYST_KEYWORDS.items():
        hits = sum(1 for word in words if word in text)
        if hits:
            tags.append({"tag": tag, "hits": hits})
    return sorted(tags, key=lambda x: x["hits"], reverse=True)


def indicator_verdict(latest: pd.Series, market_row: pd.Series | None, rel_6m: float | None):
    close = safe_float(latest.get("Close"))
    sma20 = safe_float(latest.get("SMA20"))
    sma50 = safe_float(latest.get("SMA50"))
    sma200 = safe_float(latest.get("SMA200"))
    rsi = safe_float(latest.get("RSI14"))
    macd = safe_float(latest.get("MACD"))
    macd_sig = safe_float(latest.get("MACD_SIGNAL"))
    vol = safe_float(latest.get("Volume"))
    vol20 = safe_float(latest.get("VOL20"))
    dd_1y = safe_float(latest.get("DD_1Y"))

    checks = []
    score = 0

    def add(name, state, detail, weight):
        nonlocal score
        score += weight
        checks.append({"name": name, "state": state, "detail": detail, "weight": weight})

    add("Trend", "Bullish" if close and sma200 and close > sma200 else "Bearish",
        "Price is above the 200-day moving average." if close and sma200 and close > sma200 else "Price is below the 200-day moving average.",
        18 if close and sma200 and close > sma200 else -18)
    add("Intermediate trend", "Bullish" if sma50 and sma200 and sma50 > sma200 else "Bearish",
        "50DMA is above 200DMA." if sma50 and sma200 and sma50 > sma200 else "50DMA is below 200DMA.",
        12 if sma50 and sma200 and sma50 > sma200 else -12)
    add("Momentum", "Bullish" if macd and macd_sig and macd > macd_sig else "Bearish",
        "MACD is above signal." if macd and macd_sig and macd > macd_sig else "MACD is below signal.",
        10 if macd and macd_sig and macd > macd_sig else -10)
    if rsi is not None:
        if 45 <= rsi <= 70:
            add("RSI", "Healthy", f"RSI is {rsi:.1f}, positive but not overbought.", 10)
        elif rsi > 70:
            add("RSI", "Stretched", f"RSI is {rsi:.1f}, momentum is hot and pullback risk rises.", -3)
        elif rsi < 35:
            add("RSI", "Washed out", f"RSI is {rsi:.1f}, selling pressure is high.", -8)
        else:
            add("RSI", "Neutral", f"RSI is {rsi:.1f}.", 0)
    if vol and vol20:
        add("Volume", "Confirmed" if vol > vol20 else "Quiet",
            "Volume is above 20-day average." if vol > vol20 else "Volume is below 20-day average.",
            4 if vol > vol20 else 0)
    if rel_6m is not None:
        add("Relative strength", "Leader" if rel_6m > 0 else "Laggard",
            f"6-month return versus SPY is {rel_6m * 100:.1f} percentage points.",
            14 if rel_6m > 0 else -10)
    if dd_1y is not None:
        if dd_1y > -0.10:
            add("Price damage", "Contained", f"Drawdown from 1-year high is {dd_1y * 100:.1f}%.", 8)
        elif dd_1y < -0.35:
            add("Price damage", "Severe", f"Drawdown from 1-year high is {dd_1y * 100:.1f}%.", -12)
        else:
            add("Price damage", "Repairing", f"Drawdown from 1-year high is {dd_1y * 100:.1f}%.", -4)

    if market_row is not None:
        vix = safe_float(market_row.get("vix"))
        spy_above_200 = bool(market_row.get("spy_above_200"))
        if vix is not None:
            add("Macro risk", "Supportive" if vix < 22 else "Risk-off",
                f"VIX is {vix:.1f}." + (" Fear is contained." if vix < 22 else " Fear is elevated."),
                10 if vix < 22 else -15)
        add("Market trend", "Supportive" if spy_above_200 else "Headwind",
            "SPY is above its 200DMA." if spy_above_200 else "SPY is below its 200DMA.",
            10 if spy_above_200 else -12)

    score = max(-100, min(100, score))
    if score >= 45:
        verdict = "Bullish"
    elif score >= 15:
        verdict = "Constructive"
    elif score <= -35:
        verdict = "Bearish"
    elif score <= -10:
        verdict = "Caution"
    else:
        verdict = "Neutral"
    return verdict, score, checks


def market_context():
    cache_key = "market_context"
    cached = _global_mem_cache.get(cache_key, 60)
    if cached is not None:
        return cached
    with _yf_lock:
        data = yf.download(["SPY", "QQQ", "^GSPTSE", "^VIX", "TLT", "UUP", "USO", "GLD", "SLV"], period="1y", auto_adjust=True, progress=False, threads=False)
    close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data
    spy = add_indicators(close[["SPY"]].rename(columns={"SPY": "Close"}).assign(Volume=0))
    last = spy.iloc[-1]

    def last_pct(symbol: str, periods: int = 21):
        if symbol not in close:
            return None
        series = close[symbol].dropna()
        if len(series) <= periods:
            return None
        return pct(series.pct_change(periods, fill_method=None).iloc[-1])

    def last_value(symbol: str):
        if symbol not in close:
            return None
        series = close[symbol].dropna()
        return safe_float(series.iloc[-1]) if len(series) else None

    ctx = {
        "vix": last_value("^VIX"),
        "spy_1m": last_pct("SPY"),
        "qqq_1m": last_pct("QQQ"),
        "tsx_1m": last_pct("^GSPTSE"),
        "tlt_1m": last_pct("TLT"),
        "dollar_1m": last_pct("UUP"),
        "oil_1m": last_pct("USO"),
        "gold_1m": last_pct("GLD"),
        "silver_1m": last_pct("SLV"),
        "spy_above_200": bool(last["Close"] > last["SMA200"]),
    }
    _global_mem_cache.set(cache_key, ctx, 60)
    return ctx


def alpaca_headers() -> dict:
    key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_SECRET_KEY")
    if not key or not secret:
        return {}
    return {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
    }


def alpaca_cli_cached_quote(ticker: str) -> dict | None:
    if not ALPACA_CLI_CACHE.exists():
        return None
    try:
        cache = json.loads(ALPACA_CLI_CACHE.read_text(encoding="utf-8"))
        quote = (cache.get("symbols") or {}).get(ticker)
        if not quote:
            return None
        if quote.get("error"):
            return quote
        return {
            "enabled": True,
            "source": quote.get("source", "Alpaca CLI latest trade + quote"),
            "feed": quote.get("feed", "IEX/free"),
            "last": safe_float(quote.get("last")),
            "lastSize": safe_float(quote.get("lastSize")),
            "lastExchange": quote.get("lastExchange"),
            "bid": safe_float(quote.get("bid")),
            "ask": safe_float(quote.get("ask")),
            "mid": safe_float(quote.get("mid")),
            "bidSize": safe_float(quote.get("bidSize")),
            "askSize": safe_float(quote.get("askSize")),
            "timestamp": quote.get("lastTimestamp") or quote.get("quoteTimestamp"),
            "cacheUpdated": quote.get("updated") or cache.get("updated"),
        }
    except Exception as exc:
        return {"enabled": True, "source": "Alpaca CLI cache", "error": str(exc)}


def run_alpaca_cli_json(args: list[str]) -> dict:
    env = os.environ.copy()
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                env[key.strip()] = value.strip().strip('"')
    proc = subprocess.run(
        [str(ALPACA_EXE), *args],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=18,
    )
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout).strip())
    return json.loads(proc.stdout)


def alpaca_cli_quote_once(ticker: str) -> dict | None:
    if not (ALPACA_EXE.exists() or shutil.which(str(ALPACA_EXE))):
        return None
    try:
        trade_payload = run_alpaca_cli_json([
            "data", "latest-trade", "--symbol", ticker, "--feed", "iex",
            "--profile", ALPACA_PROFILE, "--quiet",
        ])
        quote_payload = run_alpaca_cli_json([
            "data", "latest-quote", "--symbol", ticker, "--feed", "iex",
            "--profile", ALPACA_PROFILE, "--quiet",
        ])
        trade = trade_payload.get("trade", {})
        quote = quote_payload.get("quote", {})
        bid = safe_float(quote.get("bp"))
        ask = safe_float(quote.get("ap"))
        mid = None
        if bid is not None and ask is not None and bid > 0 and ask > 0:
            mid = (bid + ask) / 2
        return {
            "enabled": True,
            "source": "Alpaca CLI on-demand latest trade + quote",
            "feed": "IEX/free",
            "last": safe_float(trade.get("p")),
            "lastSize": safe_float(trade.get("s")),
            "lastExchange": trade.get("x"),
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "bidSize": safe_float(quote.get("bs")),
            "askSize": safe_float(quote.get("as")),
            "timestamp": trade.get("t") or quote.get("t"),
            "cacheUpdated": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as exc:
        return {"enabled": True, "source": "Alpaca CLI on-demand", "error": str(exc)}


def alpaca_latest_quote(ticker: str) -> dict:
    cache_key = f"latest_quote:{ticker}"
    cached_live = _global_mem_cache.get(cache_key, 30)
    if cached_live is not None:
        return cached_live
    res = _alpaca_latest_quote_uncached(ticker)
    _global_mem_cache.set(cache_key, res, 30)
    return res


def _alpaca_latest_quote_uncached(ticker: str) -> dict:
    if not alpaca_supported_symbol(ticker):
        return yfinance_latest_quote(ticker)
    cached = alpaca_cli_cached_quote(ticker)
    if cached and not cached.get("error"):
        return cached
    cli_quote = alpaca_cli_quote_once(ticker)
    if cli_quote and not cli_quote.get("error"):
        return cli_quote
    headers = alpaca_headers()
    if not headers:
        try:
            with _yf_lock:
                raw_info = yf.Ticker(ticker).fast_info
                last_price = raw_info.get("last_price")
                prev_close = raw_info.get("previous_close")
                last_volume = raw_info.get("last_volume", 0)
            if last_price is not None:
                return {
                    "enabled": True,
                    "source": "yfinance (Free Live Quote)",
                    "feed": "yfinance real-time",
                    "last": float(last_price),
                    "bid": None,
                    "ask": None,
                    "mid": float(last_price),
                    "bidSize": None,
                    "askSize": None,
                    "prevClose": float(prev_close) if prev_close else None,
                    "volume": float(last_volume),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
        except Exception as exc:
            return {"enabled": False, "source": "yfinance fallback", "error": f"Alpaca credentials missing & yfinance failed: {exc}"}
        return {"enabled": False, "source": "yfinance fallback", "error": "Alpaca credentials not configured"}
    quote_url = f"https://data.alpaca.markets/v2/stocks/{ticker}/quotes/latest"
    trade_url = f"https://data.alpaca.markets/v2/stocks/{ticker}/trades/latest"
    try:
        trade_resp = requests.get(trade_url, headers=headers, params={"feed": "iex"}, timeout=12)
        if trade_resp.status_code == 403:
            trade_resp = requests.get(trade_url, headers=headers, timeout=12)
        trade_resp.raise_for_status()
        trade_payload = trade_resp.json()
        trade = trade_payload.get("trade", {})

        quote_resp = requests.get(quote_url, headers=headers, params={"feed": "iex"}, timeout=12)
        if quote_resp.status_code == 403:
            quote_resp = requests.get(quote_url, headers=headers, timeout=12)
        quote_resp.raise_for_status()
        quote_payload = quote_resp.json()
        quote = quote_payload.get("quote", {})
        bid = safe_float(quote.get("bp"))
        ask = safe_float(quote.get("ap"))
        last = safe_float(trade.get("p"))
        mid = None
        if bid is not None and ask is not None and bid > 0 and ask > 0:
            mid = (bid + ask) / 2
        return {
            "enabled": True,
            "source": trade_payload.get("symbol") and "Alpaca latest trade + quote",
            "feed": "IEX/free" if trade_resp.url and "feed=iex" in trade_resp.url else "Alpaca market data",
            "last": last,
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "bidSize": safe_float(quote.get("bs")),
            "askSize": safe_float(quote.get("as")),
            "exchangeBid": quote.get("bx"),
            "exchangeAsk": quote.get("ax"),
            "timestamp": trade.get("t") or quote.get("t"),
        }
    except Exception as exc:
        try:
            with _yf_lock:
                raw_info = yf.Ticker(ticker).fast_info
                last_price = raw_info.get("last_price")
                prev_close = raw_info.get("previous_close")
                last_volume = raw_info.get("last_volume", 0)
            if last_price is not None:
                return {
                    "enabled": True,
                    "source": "yfinance (Free Live Quote) Fallback",
                    "feed": "yfinance real-time",
                    "last": float(last_price),
                    "bid": None,
                    "ask": None,
                    "mid": float(last_price),
                    "bidSize": None,
                    "askSize": None,
                    "prevClose": float(prev_close) if prev_close else None,
                    "volume": float(last_volume),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
        except Exception:
            pass
        return {"enabled": True, "source": "yfinance fallback", "error": str(exc)}


def alpaca_supported_symbol(ticker: str) -> bool:
    return not any(token in ticker for token in (".TO", ".V", ".NE", "=F", "^"))


def yfinance_latest_quote(ticker: str) -> dict:
    try:
        fast = yf.Ticker(ticker).fast_info
        last = safe_float(getattr(fast, "last_price", None) or fast.get("last_price"))
        previous = safe_float(getattr(fast, "previous_close", None) or fast.get("previous_close"))
        return {
            "enabled": True,
            "source": "Yahoo Finance live/delayed quote",
            "feed": "Yahoo Finance",
            "last": last,
            "mid": last,
            "previousClose": previous,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as exc:
        return {"enabled": True, "source": "yfinance fallback", "error": str(exc)}


def cycle_phase_model(score: float, sentiment_score: int, market: dict, latest: pd.Series, checks: list[dict], catalysts: list[dict]):
    vix = safe_float(market.get("vix"), 20)
    dd = safe_float(latest.get("DD_1Y"), 0)
    rsi = safe_float(latest.get("RSI14"), 50)
    ret_1m = safe_float(latest.get("RET_1M"), 0)
    ret_3m = safe_float(latest.get("RET_3M"), 0)
    ret_6m = safe_float(latest.get("RET_6M"), 0)
    close = safe_float(latest.get("Close"), 0)
    sma50 = safe_float(latest.get("SMA50"), 0)
    sma200 = safe_float(latest.get("SMA200"), 0)
    macd = safe_float(latest.get("MACD"), 0)
    macd_sig = safe_float(latest.get("MACD_SIGNAL"), 0)
    vol = safe_float(latest.get("Volume"), 0)
    vol20 = safe_float(latest.get("VOL20"), 0)
    spy_support = 1 if market.get("spy_above_200") else -1

    catalyst_names = {c["tag"] for c in catalysts}
    issuance_or_reg_risk = bool({"IPO / issuance", "Government / regulation", "Tariff / geopolitics"} & catalyst_names)
    earnings_ai_tailwind = bool({"Earnings / guidance", "AI / compute", "M&A / deal"} & catalyst_names)
    above_200 = close and sma200 and close > sma200
    above_50 = close and sma50 and close > sma50
    golden = sma50 and sma200 and sma50 > sma200
    momentum_up = macd and macd_sig and macd > macd_sig
    high_volume = vol and vol20 and vol > vol20 * 1.15

    phase_scores = {
        "Bull Stage 1 - Reset Bounce": 10,
        "Bull Stage 2 - Leadership Rotation": 10,
        "Bull Stage 3 - Euphoria / Distribution": 10,
        "Bear Stage 1 - Buy-the-Dip Trap": 10,
        "Bear Stage 2 - Grinding De-rating": 10,
        "Bear Stage 3 - Capitulation / Bottoming": 10,
    }

    phase_scores["Bull Stage 1 - Reset Bounce"] += (
        (18 if dd and dd < -0.18 else 0)
        + (14 if ret_1m and ret_1m > 0.08 else 0)
        + (10 if rsi and 35 <= rsi <= 60 else 0)
        + (10 if vix and vix > 20 else 0)
        + (8 if above_50 else 0)
        - (10 if golden and ret_6m and ret_6m > 0.25 else 0)
    )
    phase_scores["Bull Stage 2 - Leadership Rotation"] += (
        (18 if above_200 else -8)
        + (16 if golden else 0)
        + (12 if momentum_up else -4)
        + (10 if ret_3m and ret_3m > 0.05 else 0)
        + (10 if ret_6m and ret_6m > 0.08 else 0)
        + (8 if 45 <= (rsi or 0) <= 70 else 0)
        + (8 if vix and vix < 22 else -5)
        + (6 if earnings_ai_tailwind else 0)
        + (6 if spy_support > 0 else -6)
    )
    phase_scores["Bull Stage 3 - Euphoria / Distribution"] += (
        (15 if ret_6m and ret_6m > 0.35 else 0)
        + (14 if rsi and rsi > 70 else 0)
        + (10 if vix and vix < 16 else 0)
        + (10 if close and sma200 and close > sma200 * 1.25 else 0)
        + (8 if high_volume and ret_1m and ret_1m > 0.10 else 0)
        + (8 if issuance_or_reg_risk else 0)
        + (5 if sentiment_score > 2 else 0)
    )
    phase_scores["Bear Stage 1 - Buy-the-Dip Trap"] += (
        (16 if dd and -0.20 < dd < -0.06 else 0)
        + (14 if not above_50 else 0)
        + (12 if above_200 else 0)
        + (8 if vix and vix < 22 else 0)
        + (8 if ret_1m and ret_1m < -0.03 else 0)
        + (6 if sentiment_score >= 0 else 0)
        - (8 if dd and dd < -0.30 else 0)
    )
    phase_scores["Bear Stage 2 - Grinding De-rating"] += (
        (18 if not above_200 else -6)
        + (14 if not golden else 0)
        + (12 if dd and -0.35 < dd <= -0.12 else 0)
        + (10 if ret_3m and ret_3m < -0.08 else 0)
        + (8 if not momentum_up else 0)
        + (6 if sentiment_score < 0 else 0)
    )
    phase_scores["Bear Stage 3 - Capitulation / Bottoming"] += (
        (18 if dd and dd < -0.30 else 0)
        + (16 if vix and vix > 25 else 0)
        + (10 if rsi and rsi < 35 else 0)
        + (10 if high_volume and ret_1m and ret_1m < -0.08 else 0)
        + (8 if sentiment_score <= -2 else 0)
        + (8 if ret_1m and ret_1m > 0 and dd and dd < -0.25 else 0)
    )

    vals = np.array([max(1, phase_scores[k]) for k in phase_scores], dtype=float)
    vals = np.clip(vals, 1, None)
    probs = vals / vals.sum()
    rows = []
    for i, label in enumerate(phase_scores):
        rows.append({"phase": label, "probability": round(float(probs[i] * 100), 1), "score": round(float(phase_scores[label]), 1)})
    rows = sorted(rows, key=lambda x: x["probability"], reverse=True)
    current = rows[0]["phase"]
    playbooks = {
        "Bull Stage 1 - Reset Bounce": "Early rebound after damage. Avoid chasing low-quality bounce names; look for leaders that held up during the prior selloff.",
        "Bull Stage 2 - Leadership Rotation": "Best stock-picking zone. Own the strongest themes, monitor earnings reactions, and rotate as leadership passes between sectors.",
        "Bull Stage 3 - Euphoria / Distribution": "Risk of top formation. Watch insider selling, IPO/speculative issuance, crowded sentiment, and stocks rising together indiscriminately.",
        "Bear Stage 1 - Buy-the-Dip Trap": "First correction after a bull run. Confidence remains high, so dip-buying can be dangerous if trend keeps weakening.",
        "Bear Stage 2 - Grinding De-rating": "Painful middle phase. Good news may be ignored, valuations compress, and cash/defense matters more.",
        "Bear Stage 3 - Capitulation / Bottoming": "Final fear phase. Look for severe price damage, VIX spike, breadth washout, bad sentiment, and a reversal day.",
    }
    evidence = [
        f"Trend: price {'above' if above_200 else 'below'} 200DMA; 50DMA {'above' if golden else 'below'} 200DMA.",
        f"Momentum: RSI {rsi:.1f}, MACD {'above' if momentum_up else 'below'} signal.",
        f"Damage: 1-year drawdown {dd * 100:.1f}%; 1M return {ret_1m * 100:.1f}%; 6M return {ret_6m * 100:.1f}%.",
        f"Macro: VIX {vix:.1f}; SPY is {'above' if market.get('spy_above_200') else 'below'} 200DMA.",
    ]
    if catalysts:
        evidence.append("Catalysts detected: " + ", ".join(c["tag"] for c in catalysts[:4]) + ".")
    return {"current": current, "playbook": playbooks[current], "evidence": evidence, "probabilities": rows}


def chart_payload(df: pd.DataFrame, ticker: str):
    recent = df.copy()
    dates = [d.strftime("%Y-%m-%d") for d in recent.index]
    recent["DD"] = recent["Close"] / recent["Close"].cummax() - 1
    return {
        "ticker": ticker,
        "dates": dates,
        "open": recent["Open"].round(2).replace({np.nan: None}).tolist(),
        "high": recent["High"].round(2).replace({np.nan: None}).tolist(),
        "low": recent["Low"].round(2).replace({np.nan: None}).tolist(),
        "close": recent["Close"].round(2).replace({np.nan: None}).tolist(),
        "sma50": recent["SMA50"].round(2).replace({np.nan: None}).tolist(),
        "sma200": recent["SMA200"].round(2).replace({np.nan: None}).tolist(),
        "rsi": recent["RSI14"].round(1).replace({np.nan: None}).tolist(),
        "macd": recent["MACD"].round(3).replace({np.nan: None}).tolist(),
        "macdSignal": recent["MACD_SIGNAL"].round(3).replace({np.nan: None}).tolist(),
        "drawdown": recent["DD"].round(4).replace({np.nan: None}).tolist(),
        "ret1m": recent["RET_1M"].round(4).replace({np.nan: None}).tolist(),
        "volume": recent["Volume"].fillna(0).astype(float).tolist(),
    }


@app.get("/api/analyze")
def analyze():
    ticker = clean_ticker(request.args.get("ticker", "NVDA"))
    price = add_indicators(download(ticker, "5y"))
    spy = add_indicators(download("SPY", "5y"))
    market = market_context()
    latest = price.iloc[-1]
    prev = price.iloc[-2]
    rel_6m = safe_float(latest.get("RET_6M")) - safe_float(spy.iloc[-1].get("RET_6M"), 0)
    sentiment = get_consolidated_sentiment(ticker)
    live_quote = alpaca_latest_quote(ticker)
    display_price = live_quote.get("last") or live_quote.get("mid") or safe_float(latest["Close"])
    verdict, score, checks = indicator_verdict(latest, pd.Series(market), rel_6m)
    catalysts = catalyst_tags(sentiment["items"])
    phase_model = cycle_phase_model(score, sentiment["news_score"], market, latest, checks, catalysts)
    info = {}
    try:
        with _yf_lock:
            raw_info = yf.Ticker(ticker).fast_info
            info = {k: raw_info.get(k) for k in ["market_cap", "last_price", "year_high", "year_low"]}
    except Exception:
        info = {}
    response = {
        "ticker": ticker,
        "updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "price": round(safe_float(display_price), 2),
        "lastClose": round(safe_float(latest["Close"]), 2),
        "liveQuote": live_quote,
        "dailyChange": pct(latest["Close"] / prev["Close"] - 1),
        "returns": {
            "1M": pct(latest.get("RET_1M")),
            "3M": pct(latest.get("RET_3M")),
            "6M": pct(latest.get("RET_6M")),
            "1Y": pct(latest.get("RET_1Y")),
            "Drawdown": pct(latest.get("DD_1Y")),
        },
        "verdict": verdict,
        "score": round(score, 1),
        "checks": checks,
        "sentiment": sentiment,
        "catalysts": catalysts,
        "market": market,
        "phaseModel": phase_model,
        "phaseProbabilities": phase_model["probabilities"],
        "chart": chart_payload(price, ticker),
        "fastInfo": {k: safe_float(v) for k, v in info.items()},
        "notes": [
            "This tool combines technical trend, momentum, relative strength, news-title sentiment, and macro risk.",
            "It is a research workflow, not financial advice. Always verify fundamentals, earnings, valuation, and position sizing separately.",
        ],
    }
    return jsonify(response)


@app.get("/api/live")
def live():
    ticker = clean_ticker(request.args.get("ticker", "NVDA"))
    live_quote = alpaca_latest_quote(ticker)
    price = live_quote.get("last") or live_quote.get("mid") or live_quote.get("ask") or live_quote.get("bid")
    return jsonify({
        "ticker": ticker,
        "updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "price": round(safe_float(price), 2) if price is not None else None,
        "liveQuote": live_quote,
    })


@app.get("/api/markov")
def markov_api():
    ticker = clean_ticker(request.args.get("ticker", "SPY"))
    years = int(request.args.get("years", 10))
    window = int(request.args.get("window", 20))
    threshold = float(request.args.get("threshold", 0.02))
    include_hmm = request.args.get("hmm", "true").lower() != "false"
    years = max(1, min(years, 30))
    window = max(5, min(window, 252))
    threshold = max(0.001, min(threshold, 0.25))
    return jsonify(run_markov_model(ticker, years, window, threshold, include_hmm))


@app.get("/api/fusion")
def fusion_api():
    ticker = clean_ticker(request.args.get("ticker", "SPY"))
    years = int(request.args.get("years", 10))
    window = int(request.args.get("window", 20))
    threshold = float(request.args.get("threshold", 0.02))
    include_hmm = request.args.get("hmm", "false").lower() == "true"
    fast = request.args.get("fast", "true").lower() != "false"
    years = max(1, min(years, 30))
    window = max(5, min(window, 252))
    threshold = max(0.001, min(threshold, 0.25))
    return jsonify(build_fusion_payload(ticker, years, window, threshold, include_hmm=include_hmm, fast=fast))


@app.get("/")
def home():
    return render_template_string(HTML)


@app.get("/markov-skill")
def markov_skill_page():
    return render_template_string(MARKOV_PROMPT_HTML)


@app.get("/fusion")
def fusion_page():
    return render_template_string(FUSION_HTML)


HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Trade Researcher Bot - Cycle Diagnostics</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {
      --bg: #09090b;
      --panel: #141417;
      --panel2: #1e1e22;
      --line: #303036;
      --ink: #f4f4f5;
      --muted: #a1a1aa;
      --accent: #2dd4bf;
      --blue: #60a5fa;
      --amber: #f59e0b;
      --red: #fb7185;
      --green: #34d399;
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--ink); font-family: Inter, Segoe UI, Arial, sans-serif; }
    header { position: sticky; top: 0; z-index: 5; background: rgba(9,9,11,.94); border-bottom: 1px solid var(--line); backdrop-filter: blur(12px); }
    .header-inner { max-width: 1440px; margin: 0 auto; padding: 18px 22px; display: grid; grid-template-columns: minmax(280px, 1fr) auto; gap: 18px; align-items: center; }
    h1 { margin: 0; font-size: 24px; letter-spacing: 0; }
    h2 { margin: 0 0 12px; font-size: 18px; letter-spacing: 0; }
    h3 { margin: 0 0 8px; font-size: 13px; letter-spacing: 0; color: var(--muted); text-transform: uppercase; }
    p { margin: 8px 0; line-height: 1.5; color: var(--muted); }
    main { max-width: 1440px; margin: 0 auto; padding: 18px 22px 36px; }
    form { display: flex; gap: 8px; }
    input { width: 190px; border: 1px solid var(--line); border-radius: 8px; background: #0f0f12; color: var(--ink); padding: 11px 12px; font-size: 15px; text-transform: uppercase; }
    button { border: 0; border-radius: 8px; padding: 11px 14px; color: #06221e; background: var(--accent); font-weight: 760; cursor: pointer; }
    .grid { display: grid; gap: 14px; }
    .top { grid-template-columns: 1.15fr .85fr; }
    .cards { grid-template-columns: repeat(5, minmax(0, 1fr)); }
    .two { grid-template-columns: minmax(0, 1.15fr) minmax(360px, .85fr); }
    .three { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    .card { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 15px; box-shadow: 0 14px 30px rgba(0,0,0,.24); }
    .metric { font-size: 28px; font-weight: 800; }
    .sub { color: var(--muted); font-size: 13px; }
    .pill { display: inline-flex; align-items: center; gap: 6px; border: 1px solid var(--line); background: var(--panel2); color: var(--ink); border-radius: 999px; padding: 6px 9px; margin: 3px 4px 3px 0; font-size: 13px; }
    .good { color: var(--green); }
    .bad { color: var(--red); }
    .warn { color: var(--amber); }
    .accent { color: var(--accent); }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { border-bottom: 1px solid var(--line); padding: 9px 8px; text-align: left; vertical-align: top; }
    th { background: var(--panel2); color: #fafafa; }
    tr:last-child td { border-bottom: 0; }
    a { color: var(--blue); text-decoration: none; }
    .bar-wrap { height: 9px; background: #27272a; border-radius: 99px; overflow: hidden; margin-top: 5px; }
    .bar { height: 100%; background: var(--accent); width: 0%; }
    .cheat-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .phase-title { color: var(--ink); font-weight: 760; }
    .phase-card { border: 1px solid var(--line); border-radius: 8px; background: #101014; padding: 10px; }
    .phase-card.active { border-color: var(--accent); background: #10221f; }
    .toolbar { display: flex; flex-wrap: wrap; gap: 8px; margin: 0 0 12px; }
    .toolbtn { border: 1px solid var(--line); background: var(--panel2); color: var(--ink); border-radius: 8px; padding: 7px 10px; font-weight: 650; cursor: pointer; }
    .toolbtn.active { background: var(--accent); color: #06221e; border-color: var(--accent); }
    .insight-list { margin: 0; padding-left: 17px; color: var(--muted); line-height: 1.45; }
    .loading { opacity: .65; pointer-events: none; }
    .error { border-color: var(--red); color: var(--red); }
    @media (max-width: 1050px) { .top, .two, .three, .cards, .cheat-grid, .header-inner { grid-template-columns: 1fr; } form { width: 100%; } input { width: 100%; } }
  </style>
</head>
<body>
<header>
  <div class="header-inner">
    <div>
      <h1>Trade Researcher Bot - Cycle Diagnostics</h1>
      <div class="sub">State of the art market research, sentiment analysis, cycle diagnostics, and regime transitions.</div>
      <div class="sub"><a href="/fusion">Consensus Verdict</a> · <a href="/markov-skill">Markov Regimes</a></div>
    </div>
    <form id="tickerForm">
      <input id="tickerInput" value="NVDA" aria-label="Ticker" />
      <button type="submit">Analyze</button>
    </form>
  </div>
</header>
<main id="app">
  <section class="grid cards" id="summaryCards"></section>
  <section class="grid two" style="margin-top:14px">
    <div class="card">
      <div class="toolbar">
        <button class="toolbtn active" data-range="6M">6M</button>
        <button class="toolbtn" data-range="1Y">1Y</button>
        <button class="toolbtn" data-range="2Y">2Y</button>
        <button class="toolbtn active" data-toggle="ma">Moving averages</button>
        <button class="toolbtn active" data-toggle="volume">Volume</button>
      </div>
      <div id="priceChart"></div>
    </div>
    <div class="card">
      <h2>Indicator Verdict</h2>
      <div id="verdictBox"></div>
      <h2 style="margin-top:16px">Transcript Cycle Phase</h2>
      <div id="phaseBox"></div>
    </div>
  </section>
  <section class="grid two" style="margin-top:14px">
    <div class="card">
      <div class="toolbar">
        <button class="toolbtn active" data-momentum="rsi">RSI</button>
        <button class="toolbtn active" data-momentum="macd">MACD</button>
        <button class="toolbtn active" data-momentum="drawdown">Drawdown</button>
      </div>
      <div id="momentumChart"></div>
    </div>
    <div class="card">
      <h2>Macro + Sentiment Tape</h2>
      <div id="macroBox"></div>
      <h2 style="margin-top:16px">Catalyst Tags</h2>
      <div id="catalystBox"></div>
      <h2 style="margin-top:16px">Latest News Sentiment</h2>
      <div id="newsBox"></div>
    </div>
  </section>
  <section class="grid two" style="margin-top:14px">
    <div class="card">
      <h2>Checklist: What The Dashboard Is Saying</h2>
      <div id="checksBox"></div>
    </div>
    <div class="card">
      <h2>Six-Phase Cheat Sheet From Transcript</h2>
      <div class="grid cheat-grid">
        <div><div class="phase-title">Bull Stage 1: Reset Bounce</div><p>Beaten-down names bounce hardest after bear damage. New leaders usually held up better, so they may not explode immediately.</p></div>
        <div><div class="phase-title">Bull Stage 2: Leadership Rotation</div><p>Longest and best stock-picking phase. Market narrows, earnings visibility matters, and sector leaders take turns moving.</p></div>
        <div><div class="phase-title">Bull Stage 3: Euphoria</div><p>Warning phase. IPO/issuance frenzy, insider selling, speculative low-quality names, and everything rising together.</p></div>
        <div><div class="phase-title">Bear Stage 1: Buy-the-Dip Trap</div><p>First fall after a bull run. Confidence remains high, fear is low, and investors treat it as a normal correction.</p></div>
        <div><div class="phase-title">Bear Stage 2: Grinding De-rating</div><p>Good news gets ignored, valuations compress, leadership breaks, and cash starts helping performance.</p></div>
        <div><div class="phase-title">Bear Stage 3: Capitulation</div><p>Final fear phase: severe price damage, VIX spike, breadth washout, large-cap capitulation, depressed sentiment, reversal day.</p></div>
      </div>
      <h2 style="margin-top:16px">Macro/Government/IPO News Lens</h2>
      <p>Look for fiscal spending, tariffs, regulation, antitrust, defense/energy/AI infrastructure bills, Fed policy, IPO quality, insider selling, and whether speculative issuance is being rewarded. These are qualitative inputs, so use them to confirm or challenge the indicator verdict.</p>
    </div>
  </section>
</main>
<script>
const stateClass = s => {
  const x = String(s || '').toLowerCase();
  if (x.includes('bull') || x.includes('healthy') || x.includes('leader') || x.includes('support') || x.includes('confirm') || x.includes('positive')) return 'good';
  if (x.includes('bear') || x.includes('risk') || x.includes('laggard') || x.includes('severe') || x.includes('negative')) return 'bad';
  return 'warn';
};
const fmt = v => v === null || v === undefined ? 'n/a' : v;
const pct = v => v === null || v === undefined ? 'n/a' : `${v.toFixed(1)}%`;
let currentData = null;
let liveTimer = null;
let chartPrefs = { range: '6M', ma: true, volume: true, rsi: true, macd: true, drawdown: true };
async function analyze(ticker) {
  document.body.classList.add('loading');
  try {
    const res = await fetch(`/api/analyze?ticker=${encodeURIComponent(ticker)}`);
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    render(data);
    window.currentTicker = data.ticker;
  } catch (err) {
    document.getElementById('summaryCards').innerHTML = `<div class="card error">Could not analyze ticker. ${err.message}</div>`;
  } finally {
    document.body.classList.remove('loading');
  }
}
function render(data) {
  currentData = data;
  startLiveTicker(data.ticker);
  document.getElementById('tickerInput').value = data.ticker;
  document.getElementById('summaryCards').innerHTML = `
    <div class="card"><h3>Ticker</h3><div class="metric">${data.ticker}</div><div class="sub">Updated ${data.updated}</div></div>
    <div class="card"><h3>Live Price</h3><div class="metric" id="livePrice">$${data.price}</div><div class="sub ${data.dailyChange >= 0 ? 'good' : 'bad'}">Today ${pct(data.dailyChange)} vs last close $${data.lastClose}</div><div class="sub" id="liveSource">${data.liveQuote.source || ''}</div></div>
    <div class="card"><h3>Verdict</h3><div class="metric ${stateClass(data.verdict)}">${data.verdict}</div><div class="sub">Score ${data.score}/100</div></div>
    <div class="card"><h3>Sentiment</h3><div class="metric ${stateClass(data.sentiment.label)}">${data.sentiment.label}</div><div class="sub">News score ${data.sentiment.score}</div></div>
    <div class="card"><h3>Macro</h3><div class="metric ${data.market.vix < 22 ? 'good' : 'bad'}">VIX ${data.market.vix.toFixed(1)}</div><div class="sub">SPY ${data.market.spy_above_200 ? 'above' : 'below'} 200DMA</div></div>`;

  document.getElementById('verdictBox').innerHTML = `
    <p>The verdict combines trend, momentum, RSI, volume, relative strength versus SPY, price damage, VIX, and market trend.</p>
    <p><span class="pill ${data.liveQuote.error ? 'warn' : 'good'}">${data.liveQuote.source || 'Market data'}</span>
    ${data.liveQuote.feed ? `<span class="pill">${data.liveQuote.feed}</span>` : ''}
    ${data.liveQuote.last ? `<span class="pill">Last trade $${data.liveQuote.last.toFixed(2)}</span>` : ''}
    ${data.liveQuote.bid ? `<span class="pill">Bid $${data.liveQuote.bid.toFixed(2)}</span>` : ''}
    ${data.liveQuote.ask ? `<span class="pill">Ask $${data.liveQuote.ask.toFixed(2)}</span>` : ''}
    ${data.liveQuote.cacheUpdated ? `<span class="pill">CLI cache ${new Date(data.liveQuote.cacheUpdated).toLocaleTimeString()}</span>` : ''}
    ${data.liveQuote.error ? `<span class="pill warn">Fallback: ${data.liveQuote.error.slice(0, 80)}</span>` : ''}</p>
    <span class="pill">1M ${pct(data.returns['1M'])}</span>
    <span class="pill">3M ${pct(data.returns['3M'])}</span>
    <span class="pill">6M ${pct(data.returns['6M'])}</span>
    <span class="pill">1Y ${pct(data.returns['1Y'])}</span>
    <span class="pill">Drawdown ${pct(data.returns.Drawdown)}</span>`;

  document.getElementById('phaseBox').innerHTML = `
    <div class="phase-card active">
      <h3>Current phase</h3>
      <div class="metric accent" style="font-size:20px">${data.phaseModel.current}</div>
      <p>${data.phaseModel.playbook}</p>
    </div>
    <h3 style="margin-top:14px">Evidence</h3>
    <ul class="insight-list">${data.phaseModel.evidence.map(e => `<li>${e}</li>`).join('')}</ul>
    <h3 style="margin-top:14px">Probability map</h3>
    ${data.phaseProbabilities.map(p => `
      <div class="phase-card ${p.phase === data.phaseModel.current ? 'active' : ''}" style="margin:8px 0">
        <div style="display:flex;justify-content:space-between;gap:10px"><span>${p.phase}</span><strong>${p.probability}%</strong></div>
        <div class="bar-wrap"><div class="bar" style="width:${p.probability}%"></div></div>
      </div>`).join('')}`;

  document.getElementById('macroBox').innerHTML = `
    <table><tbody>
      <tr><td>SPY 1M</td><td>${pct(data.market.spy_1m)}</td><td>Broad risk appetite</td></tr>
      <tr><td>QQQ 1M</td><td>${pct(data.market.qqq_1m)}</td><td>Growth/AI leadership</td></tr>
      <tr><td>TLT 1M</td><td>${pct(data.market.tlt_1m)}</td><td>Rates pressure proxy</td></tr>
      <tr><td>Dollar 1M</td><td>${pct(data.market.dollar_1m)}</td><td>Liquidity/FX pressure</td></tr>
      <tr><td>Oil 1M</td><td>${pct(data.market.oil_1m)}</td><td>Inflation/geopolitical pressure</td></tr>
    </tbody></table>`;

  document.getElementById('newsBox').innerHTML = data.sentiment.items.length ? data.sentiment.items.slice(0, 7).map(n => `
    <p><span class="pill ${n.score > 0 ? 'good' : n.score < 0 ? 'bad' : ''}">score ${n.score}</span>
    ${n.link ? `<a href="${n.link}" target="_blank" rel="noreferrer">${n.title}</a>` : n.title}
    <br><span class="sub">${n.publisher || ''}</span></p>`).join('') : '<p>No recent Yahoo Finance news returned for this ticker.</p>';

  document.getElementById('catalystBox').innerHTML = data.catalysts.length
    ? data.catalysts.map(c => `<span class="pill">${c.tag}: ${c.hits}</span>`).join('')
    : '<p>No obvious catalyst cluster detected in recent Yahoo Finance headlines.</p>';

  document.getElementById('checksBox').innerHTML = `<table><thead><tr><th>Indicator</th><th>State</th><th>Meaning</th></tr></thead><tbody>` +
    data.checks.map(c => `<tr><td>${c.name}</td><td class="${stateClass(c.state)}">${c.state}</td><td>${c.detail}</td></tr>`).join('') +
    `</tbody></table><p>${data.notes.join(' ')}</p>`;

  renderCharts();
}
function darkLayout(title, ytitle) {
  return {
    title, paper_bgcolor:'#141417', plot_bgcolor:'#18181b', font:{color:'#f4f4f5'},
    margin:{l:54,r:36,t:54,b:42}, height:440, hovermode:'x unified',
    dragmode:'zoom',
    yaxis:{title:ytitle, gridcolor:'#2a2a2a', zerolinecolor:'#444', fixedrange:false},
    xaxis:{
      gridcolor:'#2a2a2a',
      rangeslider:{visible:false},
      rangeselector:{buttons:[
        {count:1, label:'1M', step:'month', stepmode:'backward'},
        {count:3, label:'3M', step:'month', stepmode:'backward'},
        {count:6, label:'6M', step:'month', stepmode:'backward'},
        {count:1, label:'1Y', step:'year', stepmode:'backward'},
        {step:'all', label:'All'}
      ]}
    },
    legend:{orientation:'h', y:-0.18},
    modebar:{orientation:'v'}
  };
}
function sliceChart(c) {
  const len = c.dates.length;
  const n = chartPrefs.range === '6M' ? 126 : chartPrefs.range === '1Y' ? 252 : 520;
  const start = Math.max(0, len - n);
  const out = {};
  Object.keys(c).forEach(k => out[k] = Array.isArray(c[k]) ? c[k].slice(start) : c[k]);
  return out;
}
function renderCharts() {
  if (!currentData) return;
  const c = sliceChart(currentData.chart);
  const priceTraces = [
    {x:c.dates, y:c.close, name:'Close', type:'scatter', mode:'lines', line:{color:'#2dd4bf', width:2.4}, hovertemplate:'%{x}<br>Close: $%{y:.2f}<extra></extra>'}
  ];
  if (chartPrefs.ma) {
    priceTraces.push(
      {x:c.dates, y:c.sma50, name:'50DMA', type:'scatter', mode:'lines', line:{color:'#60a5fa', width:1.5}, hovertemplate:'%{x}<br>50DMA: $%{y:.2f}<extra></extra>'},
      {x:c.dates, y:c.sma200, name:'200DMA', type:'scatter', mode:'lines', line:{color:'#f59e0b', width:1.5}, hovertemplate:'%{x}<br>200DMA: $%{y:.2f}<extra></extra>'}
    );
  }
  if (chartPrefs.volume) {
    priceTraces.push({x:c.dates, y:c.volume, name:'Volume', type:'bar', yaxis:'y2', marker:{color:'rgba(148,163,184,.28)'}, opacity:.65, hovertemplate:'%{x}<br>Volume: %{y}<extra></extra>'});
  }
  const lastDate = c.dates[c.dates.length - 1];
  const layout = darkLayout(`${currentData.ticker} Analyst Price Workbench`, 'Price');
  layout.yaxis2 = {overlaying:'y', side:'right', showgrid:false, title:'Volume', visible:chartPrefs.volume};
  layout.shapes = [
    {type:'line', xref:'paper', x0:0, x1:1, yref:'y', y0:currentData.lastClose, y1:currentData.lastClose, line:{color:'#71717a', width:1, dash:'dot'}}
  ];
  layout.annotations = [
    {x:lastDate, y:currentData.lastClose, text:'last close', showarrow:false, xanchor:'left', font:{size:11, color:'#a1a1aa'}}
  ];
  Plotly.react('priceChart', priceTraces, layout, {displayModeBar:true, responsive:true, scrollZoom:true});

  const momTraces = [];
  if (chartPrefs.rsi) {
    momTraces.push({x:c.dates, y:c.rsi, name:'RSI 14', type:'scatter', mode:'lines', yaxis:'y', line:{color:'#a78bfa'}, hovertemplate:'%{x}<br>RSI 14: %{y:.2f}<extra></extra>'});
  }
  if (chartPrefs.macd) {
    momTraces.push(
      {x:c.dates, y:c.macd, name:'MACD', type:'scatter', mode:'lines', yaxis:'y2', line:{color:'#2dd4bf'}, hovertemplate:'%{x}<br>MACD: %{y:.2f}<extra></extra>'},
      {x:c.dates, y:c.macdSignal, name:'MACD signal', type:'scatter', mode:'lines', yaxis:'y2', line:{color:'#fb7185'}, hovertemplate:'%{x}<br>Signal: %{y:.2f}<extra></extra>'}
    );
  }
  if (chartPrefs.drawdown) {
    momTraces.push({x:c.dates, y:c.drawdown.map(v => v === null ? null : v * 100), name:'Drawdown %', type:'scatter', mode:'lines', yaxis:'y3', fill:'tozeroy', line:{color:'#f59e0b', width:1.4}, hovertemplate:'%{x}<br>Drawdown: %{y:.2f}%<extra></extra>'});
  }
  const momLayout = darkLayout('Momentum + Damage Workbench', 'RSI');
  momLayout.yaxis.range = [0, 100];
  momLayout.shapes = [
    {type:'line', xref:'paper', x0:0, x1:1, yref:'y', y0:70, y1:70, line:{color:'#fb7185', dash:'dot'}},
    {type:'line', xref:'paper', x0:0, x1:1, yref:'y', y0:30, y1:30, line:{color:'#34d399', dash:'dot'}},
    {type:'line', xref:'paper', x0:0, x1:1, yref:'y3', y0:-20, y1:-20, line:{color:'#f59e0b', dash:'dot'}},
    {type:'line', xref:'paper', x0:0, x1:1, yref:'y3', y0:-35, y1:-35, line:{color:'#fb7185', dash:'dot'}}
  ];
  momLayout.yaxis2 = {overlaying:'y', side:'right', gridcolor:'#2a2a2a', zerolinecolor:'#444', title:'MACD'};
  momLayout.yaxis3 = {anchor:'free', overlaying:'y', side:'right', position:.94, showgrid:false, title:'Drawdown %'};
  Plotly.react('momentumChart', momTraces, momLayout, {displayModeBar:true, responsive:true, scrollZoom:true});
}
async function refreshLiveOnly() {
  if (!currentData?.ticker) return;
  try {
    const res = await fetch(`/api/live?ticker=${encodeURIComponent(currentData.ticker)}`);
    if (!res.ok) return;
    const data = await res.json();
    if (data.price) {
      const priceEl = document.getElementById('livePrice');
      if (priceEl) priceEl.textContent = `$${data.price}`;
    }
    const srcEl = document.getElementById('liveSource');
    if (srcEl) {
      const t = data.liveQuote?.timestamp ? new Date(data.liveQuote.timestamp).toLocaleTimeString() : data.updated;
      srcEl.textContent = `${data.liveQuote?.source || 'Alpaca CLI'} • ${data.liveQuote?.feed || ''} • ${t}`;
    }
  } catch (e) {
  }
}
function startLiveTicker(ticker) {
  if (liveTimer) clearInterval(liveTimer);
  liveTimer = setInterval(refreshLiveOnly, 20000);
}
document.querySelectorAll('[data-range]').forEach(btn => btn.addEventListener('click', () => {
  document.querySelectorAll('[data-range]').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  chartPrefs.range = btn.dataset.range;
  renderCharts();
}));
document.querySelectorAll('[data-toggle]').forEach(btn => btn.addEventListener('click', () => {
  const key = btn.dataset.toggle;
  chartPrefs[key] = !chartPrefs[key];
  btn.classList.toggle('active', chartPrefs[key]);
  renderCharts();
}));
document.querySelectorAll('[data-momentum]').forEach(btn => btn.addEventListener('click', () => {
  const key = btn.dataset.momentum;
  chartPrefs[key] = !chartPrefs[key];
  btn.classList.toggle('active', chartPrefs[key]);
  renderCharts();
}));
document.getElementById('tickerForm').addEventListener('submit', e => {
  e.preventDefault();
  analyze(document.getElementById('tickerInput').value || 'NVDA');
});
analyze(new URLSearchParams(location.search).get('ticker') || 'NVDA');
setInterval(() => analyze(window.currentTicker || document.getElementById('tickerInput').value || 'NVDA'), 15 * 60 * 1000);
</script>
</body>
</html>
"""


MARKOV_PROMPT_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Trade Researcher Bot - Markov Regimes</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {
      --bg: #09090b;
      --panel: #141417;
      --panel2: #1e1e22;
      --line: #303036;
      --ink: #f4f4f5;
      --muted: #a1a1aa;
      --accent: #2dd4bf;
      --blue: #60a5fa;
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--ink); font-family: Inter, Segoe UI, Arial, sans-serif; }
    header { position: sticky; top: 0; z-index: 3; background: rgba(9,9,11,.94); border-bottom: 1px solid var(--line); backdrop-filter: blur(12px); }
    .header-inner { max-width: 1180px; margin: 0 auto; padding: 18px 22px; display: flex; align-items: center; justify-content: space-between; gap: 18px; }
    h1 { margin: 0; font-size: 24px; letter-spacing: 0; }
    h2 { margin: 0 0 12px; font-size: 18px; }
    h3 { margin: 0 0 8px; font-size: 13px; color: var(--muted); text-transform: uppercase; }
    .sub { color: var(--muted); font-size: 13px; margin-top: 4px; }
    main { max-width: 1180px; margin: 0 auto; padding: 20px 22px 40px; }
    .card { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 16px; box-shadow: 0 14px 30px rgba(0,0,0,.24); }
    a { color: var(--blue); text-decoration: none; }
    button { border: 0; border-radius: 8px; padding: 10px 13px; color: #06221e; background: var(--accent); font-weight: 760; cursor: pointer; }
    input, select { border: 1px solid var(--line); border-radius: 8px; background: #0f0f12; color: var(--ink); padding: 10px 11px; font-size: 14px; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { border-bottom: 1px solid var(--line); padding: 9px 8px; text-align: right; }
    th:first-child, td:first-child { text-align: left; }
    th { background: var(--panel2); color: #fafafa; }
    form { display: grid; grid-template-columns: 1fr 110px 110px 120px 150px auto; gap: 10px; align-items: end; }
    label { display: grid; gap: 5px; color: var(--muted); font-size: 12px; }
    .grid { display: grid; gap: 14px; }
    .cards { grid-template-columns: repeat(4, minmax(0, 1fr)); }
    .two { grid-template-columns: minmax(0, 1.1fr) minmax(340px, .9fr); }
    .toolbar { display: flex; gap: 10px; align-items: center; justify-content: space-between; margin-bottom: 14px; }
    .pill { display: inline-flex; border: 1px solid var(--line); background: var(--panel2); color: var(--ink); border-radius: 999px; padding: 6px 9px; font-size: 13px; }
    .metric { font-size: 28px; font-weight: 800; }
    .good { color: #34d399; }
    .bad { color: #fb7185; }
    .warn { color: #f59e0b; }
    .loading { opacity: .65; pointer-events: none; }
    @media (max-width: 900px) { .header-inner, .toolbar { flex-direction: column; align-items: flex-start; } form, .cards, .two { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
<header>
  <div class="header-inner">
    <div>
      <h1>Trade Researcher Bot - Markov Regimes</h1>
      <div class="sub">Observable Markov regimes, stationary distribution, n-step forecast, walk-forward backtest, optional HMM.</div>
    </div>
    <a class="pill" href="/">Back to diagnostics</a>
  </div>
</header>
<main>
  <section class="card">
    <form id="markovForm">
      <label>Ticker <input id="ticker" value="SPY" /></label>
      <label>Years <input id="years" type="number" min="1" max="30" value="10" /></label>
      <label>Window <input id="window" type="number" min="5" max="252" value="20" /></label>
      <label>Threshold <input id="threshold" type="number" min="0.001" max="0.25" step="0.001" value="0.02" /></label>
      <label>HMM <select id="hmm"><option value="true">Try HMM</option><option value="false">Skip HMM</option></select></label>
      <button type="submit">Run model</button>
    </form>
  </section>
  <section class="grid cards" id="summary" style="margin-top:14px"></section>
  <section class="grid two" style="margin-top:14px">
    <div class="card"><div id="regimeChart"></div></div>
    <div class="card">
      <h2>Transition Matrix</h2>
      <div id="matrix"></div>
      <h2 style="margin-top:16px">Stationary Distribution</h2>
      <div id="stationary"></div>
    </div>
  </section>
  <section class="grid two" style="margin-top:14px">
    <div class="card"><div id="equityChart"></div></div>
    <div class="card">
      <h2>Forecast + HMM</h2>
      <div id="forecast"></div>
      <h2 style="margin-top:16px">Interpretation</h2>
      <div id="interpretation"></div>
    </div>
  </section>
</main>
<script>
const states = ['Bear', 'Sideways', 'Bull'];
const colors = { Bear: '#fb7185', Sideways: '#f59e0b', Bull: '#34d399' };
const fmt = v => v === null || v === undefined || Number.isNaN(v) ? 'n/a' : v;
const pct = v => v === null || v === undefined ? 'n/a' : `${(v * 100).toFixed(2)}%`;
async function runMarkov() {
  document.body.classList.add('loading');
  try {
    const params = new URLSearchParams({
      ticker: document.getElementById('ticker').value || 'SPY',
      years: document.getElementById('years').value || '10',
      window: document.getElementById('window').value || '20',
      threshold: document.getElementById('threshold').value || '0.02',
      hmm: document.getElementById('hmm').value
    });
    const res = await fetch(`/api/markov?${params.toString()}`);
    if (!res.ok) throw new Error(await res.text());
    render(await res.json());
  } catch (err) {
    document.getElementById('summary').innerHTML = `<div class="card bad">Could not run model: ${err.message}</div>`;
  } finally {
    document.body.classList.remove('loading');
  }
}
function render(data) {
  const wf = data.walkForward;
  document.getElementById('summary').innerHTML = `
    <div class="card"><h3>Ticker</h3><div class="metric">${data.ticker}</div><div class="sub">${data.start} → ${data.end} · ${data.rows} rows</div></div>
    <div class="card"><h3>Current Regime</h3><div class="metric ${data.currentState === 'Bull' ? 'good' : data.currentState === 'Bear' ? 'bad' : 'warn'}">${data.currentState}</div><div class="sub">${data.window}-day return label</div></div>
    <div class="card"><h3>Walk-forward Sharpe</h3><div class="metric">${wf.sharpe === null ? 'n/a' : wf.sharpe.toFixed(3)}</div><div class="sub">${wf.trades} evaluated trades</div></div>
    <div class="card"><h3>Max Drawdown</h3><div class="metric bad">${wf.maxDrawdown === null ? 'n/a' : (wf.maxDrawdown * 100).toFixed(2) + '%'}</div><div class="sub">strategy equity curve</div></div>`;
  document.getElementById('matrix').innerHTML = tableFromMatrix(data.matrix);
  document.getElementById('stationary').innerHTML = distributionTable(data.stationary, data.persistence, data.sampleMix);
  document.getElementById('forecast').innerHTML = forecastHtml(data);
  document.getElementById('interpretation').innerHTML = `<ul>${data.interpretation.map(x => `<li>${x}</li>`).join('')}</ul>`;
  renderRegimeChart(data);
  renderEquityChart(data);
}
function tableFromMatrix(matrix) {
  return `<table><thead><tr><th>From \\ To</th>${states.map(s => `<th>${s}</th>`).join('')}</tr></thead><tbody>` +
    matrix.map((row, i) => `<tr><td>${states[i]}</td>${row.map(v => `<td>${(v * 100).toFixed(2)}%</td>`).join('')}</tr>`).join('') +
    `</tbody></table>`;
}
function distributionTable(stationary, persistence, sampleMix) {
  return `<table><thead><tr><th>State</th><th>Stationary</th><th>Persistence</th><th>Sample Mix</th></tr></thead><tbody>` +
    states.map((s, i) => `<tr><td>${s}</td><td>${stationary[i].probability.toFixed(2)}%</td><td>${persistence[i].probability.toFixed(2)}%</td><td>${sampleMix[i].probability.toFixed(2)}%</td></tr>`).join('') +
    `</tbody></table>`;
}
function forecastHtml(data) {
  const next = data.nextDay.map(x => `<span class="pill">${x.state}: ${x.probability.toFixed(2)}%</span>`).join(' ');
  const hmm = data.hmm.available
    ? `<p class="good">${data.hmm.message}</p><table><thead><tr><th>HMM label</th><th>State</th><th>Mean daily return</th></tr></thead><tbody>${data.hmm.states.map(s => `<tr><td>${s.label}</td><td>${s.state}</td><td>${(s.meanDailyReturn * 100).toFixed(3)}%</td></tr>`).join('')}</tbody></table>`
    : `<p class="warn">${data.hmm.message}</p>`;
  return `<h3>Next-day probabilities from current state</h3><p>${next}</p><h3>5-step forecast matrix</h3>${tableFromMatrix(data.forecast5)}<h3 style="margin-top:12px">Optional HMM</h3>${hmm}`;
}
function darkLayout(title, ytitle) {
  return { title, paper_bgcolor:'#141417', plot_bgcolor:'#18181b', font:{color:'#f4f4f5'}, hovermode:'x unified', margin:{l:54,r:30,t:54,b:42}, height:430, xaxis:{gridcolor:'#2a2a2a'}, yaxis:{title:ytitle, gridcolor:'#2a2a2a', zerolinecolor:'#444'}, legend:{orientation:'h'} };
}
function renderRegimeChart(data) {
  const traces = [{
    x: data.regimeSeries.map(r => r.date),
    y: data.regimeSeries.map(r => r.close),
    mode: 'lines',
    type: 'scatter',
    name: 'Close',
    line: { color: '#2dd4bf', width: 2 },
    hovertemplate: '%{x}<br>Close: $%{y:.2f}<extra></extra>'
  }];
  states.forEach((state, id) => {
    const pts = data.regimeSeries.filter(r => r.stateId === id);
    traces.push({
      x: pts.map(r => r.date),
      y: pts.map(r => r.close),
      mode: 'markers',
      type: 'scatter',
      name: state,
      marker: { color: colors[state], size: 4, symbol: 'circle', opacity: 0.7 },
      hovertemplate: '%{x}<br>Price: $%{y:.2f}<br>Regime: ' + state + '<extra></extra>'
    });
  });
  Plotly.react('regimeChart', traces, darkLayout(`${data.ticker} Price With Observable Regime Labels`, 'Close'), {displayModeBar:true, responsive:true, scrollZoom:true});
}
function renderEquityChart(data) {
  const wf = data.walkForward;
  Plotly.react('equityChart', [{x:wf.dates, y:wf.equity, mode:'lines', type:'scatter', name:'Walk-forward equity', line:{color:'#60a5fa', width:2}}], darkLayout('Walk-forward Strategy Equity', 'Growth of $1'), {displayModeBar:true, responsive:true, scrollZoom:true});
}
document.getElementById('markovForm').addEventListener('submit', e => {
  e.preventDefault();
  runMarkov();
});
runMarkov();
</script>
</body>
</html>
"""


FUSION_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Trade Researcher Bot - Consensus Verdict</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root { --bg:#09090b; --panel:#141417; --panel2:#1e1e22; --line:#303036; --ink:#f4f4f5; --muted:#a1a1aa; --accent:#2dd4bf; --blue:#60a5fa; --amber:#f59e0b; --red:#fb7185; --green:#34d399; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--bg); color:var(--ink); font-family:Inter, Segoe UI, Arial, sans-serif; }
    header { position:sticky; top:0; z-index:4; background:rgba(9,9,11,.94); border-bottom:1px solid var(--line); backdrop-filter:blur(12px); }
    .header-inner { max-width:1480px; margin:0 auto; padding:18px 22px; display:grid; grid-template-columns:minmax(280px,1fr) auto; gap:18px; align-items:center; }
    h1 { margin:0; font-size:24px; letter-spacing:0; }
    h2 { margin:0 0 12px; font-size:18px; }
    h3 { margin:0 0 8px; font-size:12px; text-transform:uppercase; color:var(--muted); }
    .sub, p, li { color:var(--muted); line-height:1.45; }
    main { max-width:1480px; margin:0 auto; padding:18px 22px 38px; }
    form { display:flex; gap:8px; flex-wrap:wrap; align-items:end; justify-content:flex-end; }
    label { display:grid; gap:5px; color:var(--muted); font-size:12px; }
    input { border:1px solid var(--line); border-radius:8px; background:#0f0f12; color:var(--ink); padding:10px 11px; font-size:14px; }
    button { border:0; border-radius:8px; padding:10px 13px; color:#06221e; background:var(--accent); font-weight:760; cursor:pointer; }
    a { color:var(--blue); text-decoration:none; }
    .grid { display:grid; gap:14px; }
    .cards { grid-template-columns:repeat(5,minmax(0,1fr)); }
    .two { grid-template-columns:minmax(0,1.15fr) minmax(380px,.85fr); }
    .three { grid-template-columns:repeat(3,minmax(0,1fr)); }
    .quad { grid-template-columns:repeat(3,minmax(0,1fr)); }
    .card { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:15px; box-shadow:0 14px 30px rgba(0,0,0,.24); }
    .metric { font-size:28px; font-weight:800; }
    .pill { display:inline-flex; border:1px solid var(--line); background:var(--panel2); color:var(--ink); border-radius:999px; padding:6px 9px; margin:3px 4px 3px 0; font-size:13px; }
    .good { color:var(--green); } .bad { color:var(--red); } .warn { color:var(--amber); } .accent { color:var(--accent); }
    .phase-card { border:1px solid var(--line); border-radius:8px; padding:10px; background:#101014; min-height:96px; }
    .phase-card.active { border-color:var(--accent); background:#10221f; }
    .phase-card.markov { border-color:var(--blue); }
    table { width:100%; border-collapse:collapse; font-size:13px; }
    th,td { border-bottom:1px solid var(--line); padding:8px; text-align:left; vertical-align:top; }
    th { background:var(--panel2); color:#fff; }
    .toolbar { display:flex; gap:8px; flex-wrap:wrap; margin:0 0 12px; }
    .toolbtn { border:1px solid var(--line); background:var(--panel2); color:var(--ink); border-radius:8px; padding:7px 10px; font-weight:650; cursor:pointer; }
    .toolbtn.active { background:var(--accent); color:#06221e; border-color:var(--accent); }
    .loading { opacity:.65; pointer-events:none; }
    @media (max-width:1000px){ .header-inner,.cards,.two,.three,.quad{grid-template-columns:1fr;} form{justify-content:flex-start;} }
  </style>
</head>
<body>
<header>
  <div class="header-inner">
    <div>
      <h1>Consensus Verdict Dashboard</h1>
      <div class="sub">Consensus scorecard blending cycle phases, Markov transitions, social/news sentiment, and shock-event robustness.</div>
      <div class="sub"><a href="/">Cycle Diagnostics</a> · <a href="/markov-skill">Markov Regimes</a></div>
    </div>
    <form id="fusionForm">
      <label>Ticker <input id="ticker" value="SPY"></label>
      <label>Years <input id="years" type="number" min="1" max="30" value="10"></label>
      <label>Window <input id="window" type="number" min="5" max="252" value="20"></label>
      <label>Threshold <input id="threshold" type="number" min="0.001" max="0.25" step="0.001" value="0.02"></label>
      <button type="submit">Run verdict</button>
    </form>
  </div>
</header>
<main>
  <section class="grid cards" id="summary"></section>
  <section class="grid two" style="margin-top:14px">
    <div class="card">
      <div class="toolbar">
        <button class="toolbtn active" data-window="6M">6M</button>
        <button class="toolbtn" data-window="1Y">1Y</button>
        <button class="toolbtn" data-window="2Y">2Y</button>
        <button class="toolbtn" data-window="5Y">5Y</button>
        <button class="toolbtn active" data-overlay="events">Events</button>
        <button class="toolbtn active" data-overlay="ma">MAs</button>
      </div>
      <div id="fusionChart"></div>
    </div>
    <div class="card">
      <h2>Final Verdict</h2>
      <div id="finalBox"></div>
      <h2 style="margin-top:16px">Evidence</h2>
      <div id="evidenceBox"></div>
    </div>
  </section>
  <section class="grid two" style="margin-top:14px">
    <div class="card">
      <h2>Phase Quadrants</h2>
      <div class="grid quad" id="phaseGrid"></div>
    </div>
    <div class="card">
      <h2>Markov State Quadrants</h2>
      <div class="grid three" id="stateGrid"></div>
      <h2 style="margin-top:16px">Shock Robustness</h2>
      <div id="eventBox"></div>
    </div>
  </section>
  <section class="grid two" style="margin-top:14px">
    <div class="card"><div id="equityChart"></div></div>
    <div class="card">
      <h2>Macro / Sentiment / Catalysts</h2>
      <div id="contextBox"></div>
    </div>
  </section>
</main>
<script>
let current = null;
let prefs = { window:'6M', events:true, ma:true };
const phaseNames = ['Bull Stage 1 - Reset Bounce','Bull Stage 2 - Leadership Rotation','Bull Stage 3 - Euphoria / Distribution','Bear Stage 1 - Buy-the-Dip Trap','Bear Stage 2 - Grinding De-rating','Bear Stage 3 - Capitulation / Bottoming'];
const stateNames = ['Bear','Sideways','Bull'];
const stateColor = {Bear:'#fb7185', Sideways:'#f59e0b', Bull:'#34d399'};
const pct = v => v === null || v === undefined ? 'n/a' : `${v.toFixed(2)}%`;
async function runFusion(){
  document.body.classList.add('loading');
  try{
    const params = new URLSearchParams({ticker:ticker.value||'SPY', years:years.value||'10', window:window.value||'20', threshold:threshold.value||'0.02'});
    const res = await fetch(`/api/fusion?${params.toString()}`);
    if(!res.ok) throw new Error(await res.text());
    current = await res.json();
    render();
  } catch(err){
    summary.innerHTML = `<div class="card bad">Could not run fusion model: ${err.message}</div>`;
  } finally { document.body.classList.remove('loading'); }
}
function render(){
  const t = current.transcript, m = current.markov, f = current.final;
  summary.innerHTML = `
    <div class="card"><h3>Ticker</h3><div class="metric">${current.ticker}</div><div class="sub">${current.updated}</div></div>
    <div class="card"><h3>Live Price</h3><div class="metric">$${t.price}</div><div class="sub">${t.liveQuote.source||''}</div></div>
    <div class="card"><h3>Final Score</h3><div class="metric ${f.score>=58?'good':f.score>=42?'warn':'bad'}">${f.score}/100</div><div class="sub">${f.verdict}</div></div>
    <div class="card"><h3>Transcript Phase</h3><div class="metric" style="font-size:18px">${t.phaseModel.current.replace(' - ','<br>')}</div></div>
    <div class="card"><h3>Markov State</h3><div class="metric ${m.currentState==='Bull'?'good':m.currentState==='Bear'?'bad':'warn'}">${m.currentState}</div><div class="sub">Sharpe ${m.walkForward.sharpe?.toFixed(3) ?? 'n/a'}</div></div>`;
  finalBox.innerHTML = `<div class="metric ${f.score>=58?'good':f.score>=42?'warn':'bad'}">${f.verdict}</div><p>${f.action}</p>`;
  evidenceBox.innerHTML = `<ul>${f.evidence.map(x=>`<li>${x}</li>`).join('')}</ul>`;
  phaseGrid.innerHTML = phaseNames.map(name => {
    const p = t.phaseProbabilities.find(x=>x.phase===name)?.probability ?? 0;
    return `<div class="phase-card ${name===t.phaseModel.current?'active':''}"><h3>${name.split(' - ')[0]}</h3><div>${name.split(' - ')[1]}</div><div class="metric" style="font-size:20px">${p.toFixed(1)}%</div></div>`;
  }).join('');
  stateGrid.innerHTML = stateNames.map(s => {
    const n = m.nextDay.find(x=>x.state===s)?.probability ?? 0;
    const st = m.currentState === s;
    return `<div class="phase-card markov ${st?'active':''}"><h3>${st?'Current State':'State'}</h3><div class="metric" style="color:${stateColor[s]}">${s}</div><div class="sub">Next-day probability ${n.toFixed(1)}%</div></div>`;
  }).join('');
  const ea = m.eventAdjusted;
  eventBox.innerHTML = `<p>${ea.note}</p><span class="pill">Shock-window trades: ${ea.shockTrades}</span><span class="pill">Clean trades: ${ea.cleanTrades}</span><span class="pill">Clean Sharpe: ${ea.cleanSharpe===null?'n/a':ea.cleanSharpe.toFixed(3)}</span><div style="margin-top:10px">${m.events.map(e=>`<span class="pill">${e.date}: ${e.name}</span>`).join('')}</div>`;
  contextBox.innerHTML = `<table><tbody>
    <tr><td>VIX</td><td>${t.market.vix.toFixed(1)}</td></tr>
    <tr><td>SPY trend</td><td>${t.market.spy_above_200?'Above 200DMA':'Below 200DMA'}</td></tr>
    <tr><td>Sentiment</td><td>${t.sentiment.label} (${t.sentiment.score})</td></tr>
    <tr><td>Catalysts</td><td>${t.catalysts.length?t.catalysts.map(c=>c.tag).join(', '):'None detected'}</td></tr>
    <tr><td>Max drawdown</td><td>${(m.walkForward.maxDrawdown*100).toFixed(2)}%</td></tr>
    <tr><td>Stationary mix</td><td>${m.stationary.map(x=>`${x.state} ${x.probability.toFixed(1)}%`).join(' · ')}</td></tr>
  </tbody></table>`;
  renderFusionChart();
  renderEquityChart();
}
function slice(arr){ const n = prefs.window==='6M'?126:prefs.window==='1Y'?252:prefs.window==='2Y'?520:1260; return arr.slice(Math.max(0, arr.length-n)); }
function darkLayout(title,y){ return {title,paper_bgcolor:'#141417',plot_bgcolor:'#18181b',font:{color:'#f4f4f5'},hovermode:'x unified',height:470,margin:{l:54,r:30,t:54,b:42},xaxis:{gridcolor:'#2a2a2a'},yaxis:{title:y,gridcolor:'#2a2a2a',zerolinecolor:'#444'},legend:{orientation:'h'}}; }
function eventShapes(dates, yMin, yMax){
  if(!prefs.events) return [];
  return current.markov.events.map(e=>({type:'rect',xref:'x',yref:'y',x0:e.windowStart,x1:e.windowEnd,y0:yMin,y1:yMax,fillcolor:'rgba(251,113,133,.12)',line:{width:0}}));
}
function eventAnnotations(yMax){
  if(!prefs.events) return [];
  return current.markov.events.map(e=>({x:e.date,y:yMax,text:e.type,showarrow:true,arrowhead:2,ay:-28,font:{size:10,color:'#f4f4f5'}}));
}
function renderFusionChart(){
  const c = current.transcript.chart;
  const dates = slice(c.dates), close = slice(c.close), sma50 = slice(c.sma50), sma200 = slice(c.sma200);
  const traces = [{x:dates,y:close,type:'scatter',mode:'lines',name:'Close',line:{color:'#2dd4bf',width:2.4},hovertemplate:'%{x}<br>Close: $%{y:.2f}<extra></extra>'}];
  if(prefs.ma){
    traces.push(
      {x:dates,y:sma50,type:'scatter',mode:'lines',name:'50DMA',line:{color:'#60a5fa'},hovertemplate:'%{x}<br>50DMA: $%{y:.2f}<extra></extra>'},
      {x:dates,y:sma200,type:'scatter',mode:'lines',name:'200DMA',line:{color:'#f59e0b'},hovertemplate:'%{x}<br>200DMA: $%{y:.2f}<extra></extra>'}
    );
  }
  const regime = current.markov.regimeSeries.filter(r=>dates.includes(r.date));
  stateNames.forEach((s,id)=>{
    const pts=regime.filter(r=>r.stateId===id);
    traces.push({
      x:pts.map(p=>p.date),
      y:pts.map(p=>p.close),
      type:'scatter',
      mode:'markers',
      name:`Markov ${s}`,
      marker:{color:stateColor[s],size:4,symbol:'circle',opacity:0.7},
      hovertemplate:'%{x}<br>Price: $%{y:.2f}<br>State: ' + s + '<extra></extra>'
    });
  });
  const yMin = Math.min(...close.filter(x=>x)), yMax = Math.max(...close.filter(x=>x));
  const layout = darkLayout(`${current.ticker} Final Price Workbench: Transcript Trend + Markov States + Shock Windows`,'Price');
  layout.shapes = eventShapes(dates,yMin,yMax);
  layout.annotations = eventAnnotations(yMax);
  Plotly.react('fusionChart', traces, layout, {displayModeBar:true,responsive:true,scrollZoom:true});
}
function renderEquityChart(){
  const wf = current.markov.walkForward;
  const traces = [{x:wf.dates,y:wf.equity,type:'scatter',mode:'lines',name:'Markov walk-forward equity',line:{color:'#60a5fa',width:2}}];
  const layout = darkLayout('Markov Walk-forward Equity With Event Windows','Growth of $1');
  const yMin = Math.min(...wf.equity), yMax = Math.max(...wf.equity);
  layout.shapes = eventShapes(wf.dates,yMin,yMax);
  layout.annotations = eventAnnotations(yMax);
  Plotly.react('equityChart', traces, layout, {displayModeBar:true,responsive:true,scrollZoom:true});
}
document.querySelectorAll('[data-window]').forEach(btn=>btn.addEventListener('click',()=>{document.querySelectorAll('[data-window]').forEach(b=>b.classList.remove('active'));btn.classList.add('active');prefs.window=btn.dataset.window;renderFusionChart();}));
document.querySelectorAll('[data-overlay]').forEach(btn=>btn.addEventListener('click',()=>{prefs[btn.dataset.overlay]=!prefs[btn.dataset.overlay];btn.classList.toggle('active',prefs[btn.dataset.overlay]);renderFusionChart();renderEquityChart();}));
fusionForm.addEventListener('submit',e=>{e.preventDefault();runFusion();});
runFusion();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=PORT, debug=False)
