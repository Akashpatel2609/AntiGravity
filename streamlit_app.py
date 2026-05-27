from __future__ import annotations

import os

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components

from analyst_dashboard_app import (
    build_fusion_payload,
    run_markov_model,
    MARKET_SHOCK_EVENTS,
)


ASSET_PRESETS = {
    "US Stocks / ETFs": {
        "S&P 500 ETF": "SPY",
        "Nasdaq 100 ETF": "QQQ",
        "Dow ETF": "DIA",
        "Apple": "AAPL",
        "Nvidia": "NVDA",
        "Tesla": "TSLA",
    },
    "Canada - TSX": {
        "TSX Composite Index": "^GSPTSE",
        "Royal Bank of Canada": "RY.TO",
        "Toronto-Dominion Bank": "TD.TO",
        "Shopify": "SHOP.TO",
        "Enbridge": "ENB.TO",
        "Canadian Natural Resources": "CNQ.TO",
        "Suncor Energy": "SU.TO",
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
        "S&P 500 Index": "^GSPC",
        "Nasdaq Composite": "^IXIC",
        "Nasdaq 100": "^NDX",
        "Dow Jones": "^DJI",
        "Russell 2000": "^RUT",
        "VIX": "^VIX",
        "TSX Composite": "^GSPTSE",
    },
}


st.set_page_config(
    page_title="Trade Researcher Bot",
    page_icon="TR",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_streamlit_secrets() -> None:
    for key in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "ALPACA_ENDPOINT", "ALPACA_CLI_PATH"):
        try:
            value = st.secrets.get(key)
        except Exception:
            value = None
        if value:
            os.environ[key] = str(value)


load_streamlit_secrets()


def css() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #09090b; color: #f4f4f5; }
        [data-testid="stSidebar"] { background: #111114; border-right: 1px solid #303036; }
        div[data-testid="stMetric"] {
            background: #141417;
            border: 1px solid #303036;
            border-radius: 8px;
            padding: 14px;
        }
        .card {
            background: #141417;
            border: 1px solid #303036;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 14px;
        }
        .phase-card {
            background: #101014;
            border: 1px solid #303036;
            border-radius: 8px;
            padding: 12px;
            min-height: 128px;
        }
        .phase-card.active { border-color: #2dd4bf; background: #10221f; }
        .big-verdict { font-size: 1.5rem; font-weight: 800; margin-bottom: 8px; }
        .muted { color: #a1a1aa; }
        .good { color: #34d399; font-weight: 800; }
        .warn { color: #f59e0b; font-weight: 800; }
        .bad { color: #fb7185; font-weight: 800; }
        </style>
        """,
        unsafe_allow_html=True,
    )


css()


def dark_layout(title: str, ytitle: str) -> dict:
    return dict(
        title=title,
        paper_bgcolor="#141417",
        plot_bgcolor="#18181b",
        font=dict(color="#f4f4f5"),
        hovermode="x unified",
        height=480,
        margin=dict(l=54, r=30, t=54, b=42),
        xaxis=dict(
            gridcolor="#2a2a2a",
            rangeslider=dict(visible=False),
            rangeselector=dict(
                buttons=[
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(count=5, label="5Y", step="year", stepmode="backward"),
                    dict(step="all", label="All"),
                ],
                bgcolor="#222226",
                activecolor="#2dd4bf",
                font=dict(color="#f4f4f5"),
            ),
        ),
        yaxis=dict(title=ytitle, gridcolor="#2a2a2a", zerolinecolor="#444"),
        legend=dict(orientation="h"),
    )


def shock_shapes(events: list[dict], y_min: float, y_max: float) -> list[dict]:
    return [
        dict(
            type="rect",
            xref="x",
            yref="y",
            x0=event["windowStart"],
            x1=event["windowEnd"],
            y0=y_min,
            y1=y_max,
            fillcolor="rgba(251,113,133,.12)",
            line=dict(width=0),
        )
        for event in events
    ]


def shock_annotations(events: list[dict], y_max: float) -> list[dict]:
    return [
        dict(
            x=event["date"],
            y=y_max,
            text=event["type"],
            showarrow=True,
            arrowhead=2,
            ay=-28,
            font=dict(size=10, color="#f4f4f5"),
        )
        for event in events
    ]


def window_size(label: str, total: int) -> int:
    sizes = {"3M": 63, "6M": 126, "1Y": 252, "2Y": 520, "5Y": 1260, "10Y": 2520, "20Y": 5040, "30Y": 7560}
    return min(sizes.get(label, total), total)


def price_fusion_chart(
    payload: dict,
    window_label: str,
    show_events: bool,
    show_ma: bool,
    log_scale: bool,
) -> go.Figure:
    chart = payload["transcript"]["chart"]
    markov = payload["markov"]
    n = window_size(window_label, len(chart["dates"]))
    dates = chart["dates"][-n:]
    close = chart["close"][-n:]
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.035,
        row_heights=[0.56, 0.16, 0.14, 0.14],
        specs=[[{"secondary_y": False}], [{}], [{}], [{}]],
    )
    fig.add_trace(
        go.Candlestick(
            x=dates,
            open=chart["open"][-n:],
            high=chart["high"][-n:],
            low=chart["low"][-n:],
            close=close,
            name="OHLC",
            increasing_line_color="#34d399",
            decreasing_line_color="#fb7185",
            increasing_fillcolor="#34d399",
            decreasing_fillcolor="#fb7185",
        ),
        row=1,
        col=1,
    )
    if show_ma:
        fig.add_trace(go.Scatter(x=dates, y=chart["sma50"][-n:], mode="lines", name="50DMA", line=dict(color="#60a5fa"), hovertemplate="Date: %{x}<br>50DMA: $%{y:.2f}<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=dates, y=chart["sma200"][-n:], mode="lines", name="200DMA", line=dict(color="#f59e0b"), hovertemplate="Date: %{x}<br>200DMA: $%{y:.2f}<extra></extra>"), row=1, col=1)
    state_colors = {"Bear": "#fb7185", "Sideways": "#f59e0b", "Bull": "#34d399"}
    regime = [r for r in markov["regimeSeries"] if r["date"] in set(dates)]
    for state in ("Bear", "Sideways", "Bull"):
        pts = [r for r in regime if r["state"] == state]
        fig.add_trace(
            go.Scatter(
                x=[p["date"] for p in pts],
                y=[p["close"] for p in pts],
                mode="markers",
                name=f"Markov {state}",
                marker=dict(color=state_colors[state], size=4, symbol="circle", opacity=0.7),
                hovertemplate="Date: %{x}<br>Price: $%{y:.2f}<br>Regime: " + state + "<extra></extra>"
            ),
            row=1,
            col=1,
        )
    volume_colors = [
        "#34d399" if (chart["close"][-n:][i] or 0) >= (chart["open"][-n:][i] or 0) else "#fb7185"
        for i in range(len(dates))
    ]
    fig.add_trace(go.Bar(x=dates, y=chart["volume"][-n:], name="Volume", marker_color=volume_colors, hovertemplate="Date: %{x}<br>Volume: %{y}<extra></extra>"), row=2, col=1)
    fig.add_trace(go.Scatter(x=dates, y=chart["rsi"][-n:], mode="lines", name="RSI 14", line=dict(color="#a78bfa"), hovertemplate="Date: %{x}<br>RSI 14: %{y:.2f}<extra></extra>"), row=3, col=1)
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(251,113,133,.12)", line_width=0, row=3, col=1)
    fig.add_hrect(y0=0, y1=30, fillcolor="rgba(52,211,153,.10)", line_width=0, row=3, col=1)
    fig.add_trace(
        go.Scatter(x=dates, y=[x * 100 if x is not None else None for x in chart["drawdown"][-n:]], mode="lines", name="Drawdown", line=dict(color="#f97316"), hovertemplate="Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>"),
        row=4,
        col=1,
    )
    live = payload["transcript"]["liveQuote"]
    live_price = live.get("last") or live.get("mid")
    if live_price:
        fig.add_hline(
            y=live_price,
            line_dash="dot",
            line_color="#f8fafc",
            annotation_text=f"Live {live_price:.2f}",
            annotation_position="top left",
            row=1,
            col=1,
        )
    y_values = [x for x in close if isinstance(x, (int, float))]
    layout = dark_layout(f"{payload['ticker']} Final Price Workbench", "Price")
    layout["height"] = 760
    if show_events and y_values:
        layout["shapes"] = shock_shapes(markov["events"], min(y_values), max(y_values))
        layout["annotations"] = shock_annotations(markov["events"], max(y_values))
    fig.update_layout(**layout)
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_yaxes(title_text="Price", type="log" if log_scale else "linear", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)
    fig.update_yaxes(title_text="Drawdown %", row=4, col=1)
    return fig


def equity_chart(payload: dict, show_events: bool) -> go.Figure:
    walk = payload["markov"]["walkForward"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=walk["dates"],
            y=walk["equity"],
            mode="lines",
            name="Walk-forward equity",
            line=dict(color="#60a5fa", width=2),
        )
    )
    layout = dark_layout("Markov Walk-forward Equity", "Growth of $1")
    if show_events and walk["equity"]:
        layout["shapes"] = shock_shapes(payload["markov"]["events"], min(walk["equity"]), max(walk["equity"]))
        layout["annotations"] = shock_annotations(payload["markov"]["events"], max(walk["equity"]))
    fig.update_layout(**layout)
    return fig


def matrix_df(matrix: list[list[float]]) -> pd.DataFrame:
    df = pd.DataFrame(matrix, index=["Bear", "Sideways", "Bull"], columns=["Bear", "Sideways", "Bull"])
    return df.map(lambda x: f"{x * 100:.2f}%")


def fmt_number(value: float | None, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:.{digits}f}"


def fmt_percent(value: float | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value * 100:.{digits}f}%"


def normalize_symbol(raw: str, universe: str) -> str:
    symbol = (raw or "").strip().upper()
    if universe == "Canada - TSX" and symbol and "." not in symbol and not symbol.startswith("^"):
        return f"{symbol}.TO"
    return symbol or "SPY"


@st.cache_data(ttl=120, show_spinner=False)
def cached_fusion(ticker: str, years: int, window: int, threshold: float, include_hmm: bool) -> dict:
    return build_fusion_payload(ticker, years=years, window=window, threshold=threshold, include_hmm=include_hmm)


@st.cache_data(ttl=900, show_spinner=False)
def cached_markov(ticker: str, years: int, window: int, threshold: float, hmm: bool) -> dict:
    return run_markov_model(ticker, years=years, window=window, threshold=threshold, include_hmm=hmm)


@st.cache_data(ttl=300, show_spinner=False)
def cached_watchlist_item(ticker: str, window: int, threshold: float) -> dict:
    return build_fusion_payload(ticker, years=2, window=window, threshold=threshold, include_hmm=False)
def get_forecast_state_prob(curr_state_id: int, forecast_matrix: list[list[float]]) -> tuple[str, float]:
    probs = forecast_matrix[curr_state_id]
    max_val = max(probs)
    max_idx = probs.index(max_val)
    states = ["Bear", "Sideways", "Bull"]
    return states[max_idx], max_val * 100


WATCHLIST_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "watchlist.json")


def load_watchlist() -> list[str]:
    import json
    if os.path.exists(WATCHLIST_FILE):
        try:
            with open(WATCHLIST_FILE, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return [str(x).upper() for x in data]
        except Exception:
            pass
    return ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA"]


def save_watchlist(watchlist: list[str]) -> None:
    import json
    try:
        with open(WATCHLIST_FILE, "w") as f:
            json.dump(list(dict.fromkeys(watchlist)), f)
    except Exception:
        pass


# Query parameters for ticker and page selection
url_ticker = st.query_params.get("ticker", "SPY").strip().upper()
url_page = st.query_params.get("page", "Final Verdict")

st.sidebar.title("Trade Researcher Bot")

# Set session state defaults from URL query parameters
if "page_radio" not in st.session_state:
    st.session_state["page_radio"] = url_page
if "ticker_input" not in st.session_state:
    st.session_state["ticker_input"] = url_ticker

page = st.sidebar.radio("Page", ["Final Verdict", "Markov Model", "Watchlist Matrix", "Cycle Cheatsheet"], key="page_radio")
universe = st.sidebar.selectbox("Market universe", list(ASSET_PRESETS.keys()))
preset_label = st.sidebar.selectbox("Quick asset", ["Custom"] + list(ASSET_PRESETS[universe].keys()))
default_symbol = ASSET_PRESETS[universe].get(preset_label, "SPY")
if preset_label == "Custom":
    ticker = normalize_symbol(st.sidebar.text_input("Ticker", key="ticker_input"), universe)
else:
    ticker = normalize_symbol(default_symbol, universe)
    st.sidebar.caption(f"Selected symbol: {ticker}")

# Keep query params in sync
st.query_params["ticker"] = ticker
st.query_params["page"] = page

years = st.sidebar.slider("History years", 1, 30, 30)
window = st.sidebar.slider("Markov rolling window", 5, 252, 20)
threshold = st.sidebar.slider("Regime threshold", 0.001, 0.25, 0.02, 0.001)
show_events = st.sidebar.toggle("Show shock events", True)
show_ma = st.sidebar.toggle("Show moving averages", True)
log_scale = st.sidebar.toggle("Log price scale", False)
fast_mode = st.sidebar.toggle("Fast mode", True, help="Skips optional HMM on the main dashboard and reuses downloaded history for faster loads.")
auto_refresh = st.sidebar.toggle("Auto-refresh every 2 minutes", True)
window_label = st.sidebar.selectbox("Chart window", ["3M", "6M", "1Y", "2Y", "5Y", "10Y", "20Y", "30Y", "All"], index=8)

if auto_refresh:
    components.html("<script>setTimeout(() => window.parent.location.reload(), 120000)</script>", height=0)

st.title("Trade Researcher Bot Dashboard")
st.caption("Transcript cycle model + Markov regimes + news/social sentiment + shock-event robustness.")

if page == "Final Verdict":
    with st.spinner("Running combined dashboard..."):
        data = cached_fusion(ticker, years, window, threshold, not fast_mode)

    final = data["final"]
    transcript = data["transcript"]
    markov = data["markov"]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Ticker", data["ticker"])
    daily_change = transcript.get("dailyChange")
    price_delta = f"{daily_change:+.2f}%" if daily_change is not None else None
    c2.metric("Live price", f"${transcript['price']:.2f}", price_delta)
    c3.metric("Final score", f"{final['score']}/100")
    phase_current = transcript["phaseModel"]["current"]
    phase_short = phase_current.split(" - ")[0]
    c4.metric("Transcript phase", phase_short)
    c5.metric("Markov state", markov["currentState"])
    st.caption(
        f"Historical range: {markov['start']} to {markov['end']} | "
        f"{markov['rows']:,} daily bars | "
        f"Walk-forward Sharpe: {fmt_number(markov['walkForward'].get('sharpe'))} | "
        f"Quote source: {transcript['liveQuote'].get('source', 'yfinance fallback')} | "
        f"Quote time: {transcript['liveQuote'].get('timestamp') or 'latest daily close'} | "
        f"Last updated: {data['updated']}"
    )
    with st.expander("Macro tape: US, Canada, commodities", expanded=False):
        market = transcript["market"]
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("SPY 1M", f"{market.get('spy_1m', 'n/a')}%")
        m2.metric("QQQ 1M", f"{market.get('qqq_1m', 'n/a')}%")
        m3.metric("TSX 1M", f"{market.get('tsx_1m', 'n/a')}%")
        m4.metric("Gold 1M", f"{market.get('gold_1m', 'n/a')}%")
        m5.metric("Silver 1M", f"{market.get('silver_1m', 'n/a')}%")
        m6.metric("Oil 1M", f"{market.get('oil_1m', 'n/a')}%")

    left, right = st.columns([1.35, 0.85])
    with left:
        st.plotly_chart(price_fusion_chart(data, window_label, show_events, show_ma, log_scale), use_container_width=True)
    with right:
        st.markdown("### Final Verdict")
        st.markdown(f"<div class='card'><div class='big-verdict'>{final['verdict']}</div><p>{final['action']}</p></div>", unsafe_allow_html=True)
        st.markdown("### Evidence")
        for item in final["evidence"]:
            st.write(f"- {item}")
        
        st.markdown("### Live Sentiment & Social Buzz")
        sentiment = transcript.get("sentiment", {})
        lbl = sentiment.get("label", "Neutral")
        score_val = sentiment.get("score", 0)
        st.write(f"Consolidated Sentiment: **{lbl}** (Score: {score_val:+.1f})")
        st.write(f"- News Sentiment: {sentiment.get('news_score', 0):+.1f}")
        st.write(f"- Reddit Sentiment: {sentiment.get('reddit_score', 0):+.1f}")
        
        with st.expander("Show Sentiment Feed Detail"):
            items = sentiment.get("items", [])
            if items:
                for item in items[:10]:
                    st.markdown(f"- **{item['publisher']}**: [{item['title']}]({item['link']}) ({item.get('type', 'news')})")
            else:
                st.write("No news or social buzz found.")

    st.markdown("### Phase Quadrants")
    phase_cols = st.columns(3)
    for i, phase in enumerate(
        [
            "Bull Stage 1 - Reset Bounce",
            "Bull Stage 2 - Leadership Rotation",
            "Bull Stage 3 - Euphoria / Distribution",
            "Bear Stage 1 - Buy-the-Dip Trap",
            "Bear Stage 2 - Grinding De-rating",
            "Bear Stage 3 - Capitulation / Bottoming",
        ]
    ):
        prob = next((x["probability"] for x in transcript["phaseProbabilities"] if x["phase"] == phase), 0)
        active = " active" if phase == transcript["phaseModel"]["current"] else ""
        with phase_cols[i % 3]:
            st.markdown(
                f"<div class='phase-card{active}'><b>{phase}</b><br><span class='metric'>{prob:.1f}%</span></div>",
                unsafe_allow_html=True,
            )

    st.markdown("### Markov State Quadrants")
    state_cols = st.columns(3)
    for i, state in enumerate(["Bear", "Sideways", "Bull"]):
        prob = next((x["probability"] for x in markov["nextDay"] if x["state"] == state), 0)
        active = " active" if state == markov["currentState"] else ""
        with state_cols[i]:
            st.markdown(
                f"<div class='phase-card{active}'><b>{state}</b><br><span class='metric'>{prob:.1f}%</span><br><span class='muted'>next-day probability</span></div>",
                unsafe_allow_html=True,
            )

    left, right = st.columns([1.2, 0.8])
    with left:
        st.plotly_chart(equity_chart(data, show_events), use_container_width=True)
    with right:
        st.markdown("### Shock Robustness")
        st.write(markov["eventAdjusted"]["note"])
        st.write(f"Shock-window trades: {markov['eventAdjusted']['shockTrades']}")
        st.write(f"Clean trades: {markov['eventAdjusted']['cleanTrades']}")
        clean = markov["eventAdjusted"]["cleanSharpe"]
        st.write(f"Clean Sharpe: {clean:.3f}" if clean is not None else "Clean Sharpe: n/a")
        st.markdown("### Events")
        for event in markov["events"]:
            st.write(f"- {event['date']} - {event['name']}")

elif page == "Markov Model":
    include_hmm = st.sidebar.toggle("Try HMM", True)
    with st.spinner("Running Markov model..."):
        markov = cached_markov(ticker, years, window, threshold, include_hmm)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ticker", markov["ticker"])
    c2.metric("Current regime", markov["currentState"])
    c3.metric("Walk-forward Sharpe", fmt_number(markov["walkForward"].get("sharpe")))
    c4.metric("Max drawdown", fmt_percent(markov["walkForward"].get("maxDrawdown")))
    st.caption(
        f"Historical range: {markov['start']} to {markov['end']} | "
        f"{markov['rows']:,} daily bars | "
        f"Regime threshold: {threshold} | "
        f"Rolling window: {window} days"
    )

    st.markdown("### Transition Matrix")
    st.dataframe(matrix_df(markov["matrix"]), use_container_width=True)
    st.markdown("### Stationary Distribution")
    st.dataframe(pd.DataFrame(markov["stationary"]), use_container_width=True)

    st.plotly_chart(
        equity_chart({"markov": markov}, show_events),
        use_container_width=True,
    )
    st.markdown("### HMM")
    st.write(markov["hmm"]["message"])
    if markov["hmm"].get("states"):
        st.dataframe(pd.DataFrame(markov["hmm"]["states"]), use_container_width=True)

elif page == "Watchlist Matrix":
    st.subheader("Personal Watchlist Matrix")
    st.caption("A multi-horizon forecasting matrix of your watchlist stocks. Click on a ticker to load its detailed charts and metrics.")

    # Search & Add Stock & Refresh
    col_search, col_add, col_ref = st.columns([3, 1.2, 1.2])
    with col_search:
        new_ticker = st.text_input(
            "Search and Add Stock",
            placeholder="Search ticker (e.g. AMD, META, AMZN)...",
            label_visibility="collapsed",
            key="watchlist_new_ticker"
        ).strip().upper()
    with col_add:
        add_clicked = st.button("➕ Add Stock", use_container_width=True)
    with col_ref:
        if st.button("🔄 Refresh Data", use_container_width=True, help="Clear all caching layers and reload fresh data"):
            from analyst_dashboard_app import clear_backend_cache
            clear_backend_cache()
            st.cache_data.clear()
            st.rerun()

    if add_clicked and new_ticker:
        from analyst_dashboard_app import clean_ticker, _yf_lock
        cleaned_tk = clean_ticker(new_ticker)
        watchlist = load_watchlist()
        if cleaned_tk in watchlist:
            st.warning(f"Ticker {cleaned_tk} is already in your watchlist.")
        else:
            with st.spinner(f"Verifying {cleaned_tk}..."):
                try:
                    import yfinance as yf
                    with _yf_lock:
                        check_df = yf.download(cleaned_tk, period="5d", progress=False, threads=False)
                    if not check_df.empty:
                        watchlist.append(cleaned_tk)
                        save_watchlist(watchlist)
                        st.success(f"Added {cleaned_tk} to your watchlist!")
                        st.rerun()
                    else:
                        st.error(f"No price data found for {cleaned_tk}. Verify the symbol is correct.")
                except Exception as e:
                    st.error(f"Could not verify {cleaned_tk}: {e}")

    # Load and render watchlist
    watchlist = load_watchlist()
    if not watchlist:
        st.info("Your watchlist is empty. Search and add tickers above.")
    else:
        import concurrent.futures
        watchlist_data = []
        with st.spinner("Fetching watchlist metrics in parallel..."):
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(watchlist), 15)) as executor:
                futures = [
                    (tk, executor.submit(cached_watchlist_item, tk, window, threshold))
                    for tk in watchlist
                ]
                for tk, future in futures:
                    try:
                        data = future.result()
                        watchlist_data.append(data)
                    except Exception as e:
                        st.error(f"Error fetching data for {tk}: {e}")

        # Build forecasts table
        if watchlist_data:
            st.markdown(
                """
                <style>
                .badge-buy { background-color: #10221f; color: #2dd4bf; border: 1px solid #2dd4bf; border-radius: 4px; padding: 2px 6px; font-weight: bold; font-size: 0.85em; }
                .badge-accumulate { background-color: #1a231f; color: #34d399; border: 1px solid #34d399; border-radius: 4px; padding: 2px 6px; font-weight: bold; font-size: 0.85em; }
                .badge-hold { background-color: #221a0f; color: #f59e0b; border: 1px solid #f59e0b; border-radius: 4px; padding: 2px 6px; font-weight: bold; font-size: 0.85em; }
                .badge-reduce { background-color: #221414; color: #f97316; border: 1px solid #f97316; border-radius: 4px; padding: 2px 6px; font-weight: bold; font-size: 0.85em; }
                .badge-sell { background-color: #251215; color: #fb7185; border: 1px solid #fb7185; border-radius: 4px; padding: 2px 6px; font-weight: bold; font-size: 0.85em; }
                
                .state-bull { color: #34d399; font-weight: bold; }
                .state-sideways { color: #f59e0b; font-weight: bold; }
                .state-bear { color: #fb7185; font-weight: bold; }
                </style>
                """,
                unsafe_allow_html=True
            )

            # Table header
            t_col1, t_col2, t_col3, t_col4, t_col5, t_col6, t_col7, t_col8, t_col9, t_col10 = st.columns(
                [1.1, 1.4, 1.6, 1.2, 1.8, 1.4, 1.4, 1.4, 1.4, 0.6]
            )
            t_col1.write("**Ticker**")
            t_col2.write("**Price / Change**")
            t_col3.write("**Signal**")
            t_col4.write("**Markov**")
            t_col5.write("**Cycle Phase**")
            t_col6.write("**Tomorrow**")
            t_col7.write("**3 Days**")
            t_col8.write("**1 Week**")
            t_col9.write("**1 Month**")
            t_col10.write("**Del**")
            st.markdown("<hr style='margin: 4px 0 12px 0; border-color: #303036;'>", unsafe_allow_html=True)

            def get_action_badge(score: float) -> str:
                if score >= 72:
                    return '<span class="badge-buy">🟢 BUY</span>'
                elif score >= 58:
                    return '<span class="badge-accumulate">🟢 ACCUM</span>'
                elif score >= 42:
                    return '<span class="badge-hold">🟡 HOLD</span>'
                elif score >= 28:
                    return '<span class="badge-reduce">🟠 REDUCE</span>'
                else:
                    return '<span class="badge-sell">🔴 SELL</span>'

            def get_state_span(state: str) -> str:
                cls_map = {"Bull": "state-bull", "Sideways": "state-sideways", "Bear": "state-bear"}
                return f'<span class="{cls_map.get(state, "")}">{state}</span>'

            for row in watchlist_data:
                r_col1, r_col2, r_col3, r_col4, r_col5, r_col6, r_col7, r_col8, r_col9, r_col10 = st.columns(
                    [1.1, 1.4, 1.6, 1.2, 1.8, 1.4, 1.4, 1.4, 1.4, 0.6]
                )
                ticker_symbol = row["ticker"]
                transcript = row["transcript"]
                markov = row["markov"]
                final = row["final"]

                # 1. Ticker button
                with r_col1:
                    if st.button(ticker_symbol, key=f"select_{ticker_symbol}", help=f"Show full verdict & charts for {ticker_symbol}"):
                        st.session_state["ticker_input"] = ticker_symbol
                        st.session_state["page_radio"] = "Final Verdict"
                        st.rerun()

                # 2. Price / Daily Change
                price = transcript["price"]
                change = transcript["dailyChange"]
                change_val = change if change is not None else 0.0
                change_color = "#34d399" if change_val >= 0 else "#fb7185"
                r_col2.markdown(
                    f"**${price:.2f}**<br><span style='color:{change_color}; font-size:0.85em;'>{change_val:+.2f}%</span>",
                    unsafe_allow_html=True
                )

                # 3. Buy/Sell/Hold Signal
                score = final["score"]
                r_col3.markdown(
                    f"{get_action_badge(score)}<br><span class='muted' style='font-size:0.8em;'>Score: {score}</span>",
                    unsafe_allow_html=True
                )

                # 4. Markov State
                m_state = markov["currentState"]
                r_col4.markdown(get_state_span(m_state), unsafe_allow_html=True)

                # 5. Cycle Phase
                c_phase = transcript["phaseModel"]["current"]
                phase_parts = c_phase.split(" - ")
                phase_short = phase_parts[0].replace("Bull Stage ", "B").replace("Bear Stage ", "Bear ")
                phase_detail = phase_parts[1] if len(phase_parts) > 1 else ""
                r_col5.markdown(
                    f"<span style='font-size:0.9em; font-weight:500;'>{phase_short}</span><br><span class='muted' style='font-size:0.75em;'>{phase_detail}</span>",
                    unsafe_allow_html=True
                )

                # 6. Tomorrow Prediction
                next_day = max(markov["nextDay"], key=lambda x: x["probability"])
                tom_state = next_day["state"]
                tom_prob = next_day["probability"]
                r_col6.markdown(
                    f"{get_state_span(tom_state)}<br><span class='muted' style='font-size:0.8em;'>{tom_prob:.0f}%</span>",
                    unsafe_allow_html=True
                )

                # 7. 3 Days Prediction
                f3_state, f3_prob = get_forecast_state_prob(markov["currentStateId"], markov["forecast3"])
                r_col7.markdown(
                    f"{get_state_span(f3_state)}<br><span class='muted' style='font-size:0.8em;'>{f3_prob:.0f}%</span>",
                    unsafe_allow_html=True
                )

                # 8. 1 Week Prediction
                f5_state, f5_prob = get_forecast_state_prob(markov["currentStateId"], markov["forecast5"])
                r_col8.markdown(
                    f"{get_state_span(f5_state)}<br><span class='muted' style='font-size:0.8em;'>{f5_prob:.0f}%</span>",
                    unsafe_allow_html=True
                )

                # 9. 1 Month Prediction
                f20_state, f20_prob = get_forecast_state_prob(markov["currentStateId"], markov["forecast20"])
                r_col9.markdown(
                    f"{get_state_span(f20_state)}<br><span class='muted' style='font-size:0.8em;'>{f20_prob:.0f}%</span>",
                    unsafe_allow_html=True
                )

                # 10. Delete Button
                with r_col10:
                    if st.button("❌", key=f"del_{ticker_symbol}", help=f"Remove {ticker_symbol} from watchlist"):
                        watchlist = load_watchlist()
                        if ticker_symbol in watchlist:
                            watchlist.remove(ticker_symbol)
                            save_watchlist(watchlist)
                            st.success(f"Removed {ticker_symbol}!")
                            st.rerun()
                
                st.markdown("<hr style='margin: 6px 0; border-color: #222225;'>", unsafe_allow_html=True)

else:
    st.markdown(
        """
        ### How to read this dashboard

        - **Transcript phase** answers: where are we in a six-stage bull/bear cycle?
        - **Markov state** answers: based on rolling-return labels, what regime are we in and what historically came next?
        - **Final score** blends both, then adjusts for macro risk, sentiment, drawdown, and shock-event robustness.
        - **Shock windows** mark periods around crashes, pandemics, wars, policy shocks, and credit events so you can see whether the model only worked because of unusual history.
        - This is a research tool, not financial advice.
        """
    )
    st.markdown("### Event library")
    st.dataframe(pd.DataFrame(MARKET_SHOCK_EVENTS), use_container_width=True)
