from __future__ import annotations

import json
import os
import time

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf

from analyst_dashboard_app import build_fusion_payload, chart_payload, add_indicators


ASSET_PRESETS = {
    "US Stocks / ETFs": {
        "S&P 500 ETF": "SPY",
        "Nasdaq 100 ETF": "QQQ",
        "Apple": "AAPL",
        "Nvidia": "NVDA",
        "Tesla": "TSLA",
    },
    "Canada - TSX": {
        "TSX Composite": "^GSPTSE",
        "Royal Bank": "RY.TO",
        "TD Bank": "TD.TO",
        "Shopify": "SHOP.TO",
        "Enbridge": "ENB.TO",
        "Suncor": "SU.TO",
        "Barrick Gold": "ABX.TO",
    },
    "Commodities / Macro": {
        "Gold futures": "GC=F",
        "Silver futures": "SI=F",
        "Crude oil futures": "CL=F",
        "Natural gas futures": "NG=F",
        "Gold ETF": "GLD",
        "Silver ETF": "SLV",
        "Oil ETF": "USO",
        "US Dollar ETF": "UUP",
        "20Y Treasury ETF": "TLT",
    },
    "Major Indices": {
        "S&P 500": "^GSPC",
        "Nasdaq Composite": "^IXIC",
        "Nasdaq 100": "^NDX",
        "Dow Jones": "^DJI",
        "Russell 2000": "^RUT",
        "VIX": "^VIX",
    },
}

INTERVALS = {
    "1m": ("1d", "1m"),
    "5m": ("5d", "5m"),
    "15m": ("1mo", "15m"),
    "1h": ("6mo", "60m"),
    "1D": ("2y", "1d"),
    "1W": ("10y", "1wk"),
    "1M": ("max", "1mo"),
}


st.set_page_config(page_title="Trade Researcher Bot", page_icon="TR", layout="wide", initial_sidebar_state="expanded")


def load_streamlit_secrets() -> None:
    for key in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "ALPACA_ENDPOINT", "ALPACA_CLI_PATH"):
        try:
            value = st.secrets.get(key)
        except Exception:
            value = None
        if value:
            os.environ[key] = str(value)


load_streamlit_secrets()


st.markdown(
    """
    <style>
      .stApp { background:#09090b; color:#f4f4f5; }
      [data-testid="stSidebar"] { background:#111114; border-right:1px solid #303036; }
      div[data-testid="stMetric"] { background:#141417; border:1px solid #303036; border-radius:8px; padding:12px; }
      .panel { background:#141417; border:1px solid #303036; border-radius:8px; padding:14px; margin-bottom:12px; }
      .verdict { font-size:1.35rem; font-weight:800; margin-bottom:6px; }
      .muted { color:#a1a1aa; }
      .small { font-size:.86rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def normalize_symbol(raw: str, universe: str) -> str:
    symbol = (raw or "").strip().upper()
    if universe == "Canada - TSX" and symbol and "." not in symbol and not symbol.startswith("^"):
        return f"{symbol}.TO"
    return symbol or "SPY"


def fmt_number(value: float | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:.{digits}f}"


def fmt_price(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:,.2f}"


@st.cache_data(ttl=60, show_spinner=False)
def cached_fusion(ticker: str, years: int, window: int, threshold: float) -> dict:
    return build_fusion_payload(
        ticker,
        years=years,
        window=window,
        threshold=threshold,
        include_hmm=False,
        fast=True,
    )


@st.cache_data(ttl=15, show_spinner=False)
def cached_chart(ticker: str, interval_label: str) -> dict:
    period, interval = INTERVALS[interval_label]
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False, threads=False)
    if df.empty:
        raise ValueError(f"No chart data returned for {ticker} at {interval_label}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(how="all")
    df.index = pd.to_datetime(df.index).tz_localize(None)
    if interval in ("1d", "1wk", "1mo"):
        df = add_indicators(df)
        return chart_payload(df, ticker)

    # Intraday indicators use shorter moving averages so they respond visually.
    out = df.copy()
    out["SMA50"] = out["Close"].rolling(50).mean()
    out["SMA200"] = out["Close"].rolling(200).mean()
    out["RSI14"] = None
    out["MACD"] = None
    out["MACD_SIGNAL"] = None
    out["RET_1M"] = None
    out["DD"] = out["Close"] / out["Close"].cummax() - 1
    return {
        "ticker": ticker,
        "dates": [d.strftime("%Y-%m-%d %H:%M") for d in out.index],
        "open": out["Open"].round(4).replace({pd.NA: None}).where(pd.notna(out["Open"]), None).tolist(),
        "high": out["High"].round(4).replace({pd.NA: None}).where(pd.notna(out["High"]), None).tolist(),
        "low": out["Low"].round(4).replace({pd.NA: None}).where(pd.notna(out["Low"]), None).tolist(),
        "close": out["Close"].round(4).replace({pd.NA: None}).where(pd.notna(out["Close"]), None).tolist(),
        "sma50": out["SMA50"].round(4).where(pd.notna(out["SMA50"]), None).tolist(),
        "sma200": out["SMA200"].round(4).where(pd.notna(out["SMA200"]), None).tolist(),
        "volume": out["Volume"].fillna(0).astype(float).tolist(),
    }


def lightweight_chart(chart: dict, live_price: float | None, interval_label: str) -> None:
    candles = []
    volumes = []
    ma50 = []
    ma200 = []
    for i, date in enumerate(chart["dates"]):
        if None in (chart["open"][i], chart["high"][i], chart["low"][i], chart["close"][i]):
            continue
        timestamp = int(pd.Timestamp(date).timestamp())
        item = {
            "time": timestamp,
            "open": float(chart["open"][i]),
            "high": float(chart["high"][i]),
            "low": float(chart["low"][i]),
            "close": float(chart["close"][i]),
        }
        candles.append(item)
        color = "rgba(52,211,153,.45)" if item["close"] >= item["open"] else "rgba(251,113,133,.45)"
        volumes.append({"time": timestamp, "value": float(chart["volume"][i] or 0), "color": color})
        if chart.get("sma50") and chart["sma50"][i] is not None:
            ma50.append({"time": timestamp, "value": float(chart["sma50"][i])})
        if chart.get("sma200") and chart["sma200"][i] is not None:
            ma200.append({"time": timestamp, "value": float(chart["sma200"][i])})

    html = f"""
    <div id="tvchart" style="height:620px;width:100%;"></div>
    <script src="https://unpkg.com/lightweight-charts@4.2.1/dist/lightweight-charts.standalone.production.js"></script>
    <script>
      const container = document.getElementById('tvchart');
      const chart = LightweightCharts.createChart(container, {{
        layout: {{ background: {{ color: '#141417' }}, textColor: '#f4f4f5' }},
        grid: {{ vertLines: {{ color: '#27272a' }}, horzLines: {{ color: '#27272a' }} }},
        rightPriceScale: {{ borderColor: '#3f3f46' }},
        timeScale: {{ borderColor: '#3f3f46', timeVisible: {str(interval_label in ["1m", "5m", "15m", "1h"]).lower()} }},
        crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
      }});
      const candleSeries = chart.addCandlestickSeries({{
        upColor: '#34d399',
        downColor: '#fb7185',
        borderUpColor: '#34d399',
        borderDownColor: '#fb7185',
        wickUpColor: '#34d399',
        wickDownColor: '#fb7185',
      }});
      candleSeries.setData({json.dumps(candles)});
      const volumeSeries = chart.addHistogramSeries({{
        priceFormat: {{ type: 'volume' }},
        priceScaleId: '',
        scaleMargins: {{ top: 0.82, bottom: 0 }},
      }});
      volumeSeries.setData({json.dumps(volumes)});
      const ma50 = chart.addLineSeries({{ color: '#60a5fa', lineWidth: 1, priceLineVisible: false }});
      ma50.setData({json.dumps(ma50)});
      const ma200 = chart.addLineSeries({{ color: '#f59e0b', lineWidth: 1, priceLineVisible: false }});
      ma200.setData({json.dumps(ma200)});
      {"candleSeries.createPriceLine({ price: " + str(float(live_price)) + ", color: '#f8fafc', lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dotted, axisLabelVisible: true, title: 'Live' });" if live_price else ""}
      chart.timeScale().fitContent();
      new ResizeObserver(entries => {{
        if (entries.length === 0 || entries[0].target !== container) return;
        chart.applyOptions({{ width: entries[0].contentRect.width }});
      }}).observe(container);
    </script>
    """
    components.html(html, height=640)


st.sidebar.title("Trade Researcher Bot")
universe = st.sidebar.selectbox("Market", list(ASSET_PRESETS.keys()))
preset_label = st.sidebar.selectbox("Asset", ["Custom"] + list(ASSET_PRESETS[universe].keys()))
default_symbol = ASSET_PRESETS[universe].get(preset_label, "SPY")
ticker_input = st.sidebar.text_input("Ticker", default_symbol, help="Canada examples: RY.TO, TD.TO. In Canada mode, typing RY becomes RY.TO.")
ticker = normalize_symbol(default_symbol if preset_label != "Custom" else ticker_input, universe)

interval_label = st.sidebar.radio("Interval", list(INTERVALS.keys()), index=4, horizontal=True)
years = st.sidebar.slider("Verdict history", 1, 10, 2, help="Lower history is much faster. Use longer only when researching, not live monitoring.")
window = st.sidebar.slider("Regime window", 5, 80, 20)
threshold = st.sidebar.slider("Regime threshold", 0.001, 0.08, 0.02, 0.001)
refresh_seconds = st.sidebar.selectbox("Auto-refresh", [5, 10, 15, 30, 60], index=2)

if st.sidebar.button("Refresh now", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

components.html(f"<script>setTimeout(() => window.parent.location.reload(), {refresh_seconds * 1000})</script>", height=0)

start = time.perf_counter()
with st.spinner("Refreshing live verdict..."):
    data = cached_fusion(ticker, years, window, threshold)
    chart = data["transcript"]["chart"] if interval_label == "1D" else cached_chart(ticker, interval_label)
elapsed_ms = int((time.perf_counter() - start) * 1000)

final = data["final"]
transcript = data["transcript"]
markov = data["markov"]
live_quote = transcript["liveQuote"]
live_price = live_quote.get("last") or live_quote.get("mid") or transcript.get("price")

st.title("Trade Researcher Bot")
st.caption("Single-screen fusion verdict with TradingView-style charts. Fast mode skips heavy backtests for live refresh speed.")

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Ticker", data["ticker"])
c2.metric("Price", fmt_price(live_price), f"{fmt_number(transcript.get('dailyChange'))}%")
c3.metric("Final score", f"{final['score']}/100")
c4.metric("Verdict", final["verdict"])
c5.metric("Regime", markov["currentState"])
c6.metric("Load", f"{elapsed_ms} ms")

st.caption(
    f"Quote source: {live_quote.get('source', 'unknown')} | "
    f"Quote time: {live_quote.get('timestamp') or 'latest available'} | "
    f"Chart interval: {interval_label} | "
    f"Verdict history: {markov['start']} to {markov['end']}"
)

left, right = st.columns([1.55, 0.8])
with left:
    lightweight_chart(chart, live_price, interval_label)

with right:
    st.markdown(f"<div class='panel'><div class='verdict'>{final['verdict']}</div><div>{final['action']}</div></div>", unsafe_allow_html=True)
    st.markdown("#### Evidence")
    for item in final["evidence"][:6]:
        st.write(f"- {item}")

    sentiment = transcript.get("sentiment", {})
    st.markdown("#### Sentiment")
    st.write(f"Combined: **{sentiment.get('label', 'Neutral')}** ({sentiment.get('score', 0):+.1f})")
    st.write(f"News: {sentiment.get('news_score', 0):+.1f} | Reddit: {sentiment.get('reddit_score', 0):+.1f}")

    market = transcript["market"]
    st.markdown("#### Macro Tape")
    st.write(f"SPY 1M: {market.get('spy_1m', 'n/a')}% | QQQ 1M: {market.get('qqq_1m', 'n/a')}%")
    st.write(f"TSX 1M: {market.get('tsx_1m', 'n/a')}% | Gold 1M: {market.get('gold_1m', 'n/a')}%")
    st.write(f"Silver 1M: {market.get('silver_1m', 'n/a')}% | Oil 1M: {market.get('oil_1m', 'n/a')}%")

st.markdown("#### Phase And State")
p1, p2, p3 = st.columns(3)
phase = transcript["phaseModel"]["current"]
p1.metric("Cycle phase", phase)
next_bull = next((x["probability"] for x in markov["nextDay"] if x["state"] == "Bull"), 0)
next_bear = next((x["probability"] for x in markov["nextDay"] if x["state"] == "Bear"), 0)
p2.metric("Next Bull", f"{next_bull:.1f}%")
p3.metric("Next Bear", f"{next_bear:.1f}%")

st.info(
    "Real-time note: US symbols can use Alpaca/IEX latest quote when configured. "
    "Canadian equities, futures, and most indices use Yahoo Finance live/delayed data. "
    "True tick-by-tick exchange data for all markets requires paid streaming feeds."
)
