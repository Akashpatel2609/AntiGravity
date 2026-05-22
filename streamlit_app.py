from __future__ import annotations

import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analyst_dashboard_app import (
    build_fusion_payload,
    run_markov_model,
    MARKET_SHOCK_EVENTS,
)


st.set_page_config(
    page_title="AntiGravity Stock Research",
    page_icon="AG",
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
        xaxis=dict(gridcolor="#2a2a2a"),
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


def price_fusion_chart(payload: dict, window_label: str, show_events: bool, show_ma: bool) -> go.Figure:
    chart = payload["transcript"]["chart"]
    markov = payload["markov"]
    n = {"6M": 126, "1Y": 252, "2Y": 520, "5Y": 1260}.get(window_label, 252)
    dates = chart["dates"][-n:]
    close = chart["close"][-n:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=close, mode="lines", name="Close", line=dict(color="#2dd4bf", width=2.4)))
    if show_ma:
        fig.add_trace(go.Scatter(x=dates, y=chart["sma50"][-n:], mode="lines", name="50DMA", line=dict(color="#60a5fa")))
        fig.add_trace(go.Scatter(x=dates, y=chart["sma200"][-n:], mode="lines", name="200DMA", line=dict(color="#f59e0b")))
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
                marker=dict(color=state_colors[state], size=6, symbol="square"),
            )
        )
    y_values = [x for x in close if isinstance(x, (int, float))]
    layout = dark_layout(f"{payload['ticker']} Final Price Workbench", "Price")
    if show_events and y_values:
        layout["shapes"] = shock_shapes(markov["events"], min(y_values), max(y_values))
        layout["annotations"] = shock_annotations(markov["events"], max(y_values))
    fig.update_layout(**layout)
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


@st.cache_data(ttl=900, show_spinner=False)
def cached_fusion(ticker: str, years: int, window: int, threshold: float) -> dict:
    return build_fusion_payload(ticker, years=years, window=window, threshold=threshold)


@st.cache_data(ttl=900, show_spinner=False)
def cached_markov(ticker: str, years: int, window: int, threshold: float, hmm: bool) -> dict:
    return run_markov_model(ticker, years=years, window=window, threshold=threshold, include_hmm=hmm)


st.sidebar.title("AntiGravity")
page = st.sidebar.radio("Page", ["Final Verdict", "Markov Model", "Cycle Cheatsheet"])
ticker = st.sidebar.text_input("Ticker", "SPY").strip().upper() or "SPY"
years = st.sidebar.slider("History years", 1, 30, 10)
window = st.sidebar.slider("Markov rolling window", 5, 252, 20)
threshold = st.sidebar.slider("Regime threshold", 0.001, 0.25, 0.02, 0.001)
show_events = st.sidebar.toggle("Show shock events", True)
show_ma = st.sidebar.toggle("Show moving averages", True)
window_label = st.sidebar.selectbox("Chart window", ["6M", "1Y", "2Y", "5Y"], index=1)

st.title("AntiGravity Stock Research Dashboard")
st.caption("Transcript cycle model + Markov regimes + macro/sentiment + shock-event robustness.")

if page == "Final Verdict":
    with st.spinner("Running combined dashboard..."):
        data = cached_fusion(ticker, years, window, threshold)

    final = data["final"]
    transcript = data["transcript"]
    markov = data["markov"]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Ticker", data["ticker"], data["updated"])
    c2.metric("Live price", f"${transcript['price']}", transcript["liveQuote"].get("source", ""))
    c3.metric("Final score", f"{final['score']}/100", final["verdict"])
    c4.metric("Transcript phase", transcript["phaseModel"]["current"].replace(" - ", " | "))
    c5.metric("Markov state", markov["currentState"], f"Sharpe {fmt_number(markov['walkForward'].get('sharpe'))}")

    left, right = st.columns([1.35, 0.85])
    with left:
        st.plotly_chart(price_fusion_chart(data, window_label, show_events, show_ma), use_container_width=True)
    with right:
        st.markdown("### Final Verdict")
        st.markdown(f"<div class='card'><div class='metric'>{final['verdict']}</div><p>{final['action']}</p></div>", unsafe_allow_html=True)
        st.markdown("### Evidence")
        for item in final["evidence"]:
            st.write(f"- {item}")

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
    c1.metric("Ticker", markov["ticker"], f"{markov['rows']} rows")
    c2.metric("Current regime", markov["currentState"])
    c3.metric("Walk-forward Sharpe", fmt_number(markov["walkForward"].get("sharpe")))
    c4.metric("Max drawdown", fmt_percent(markov["walkForward"].get("maxDrawdown")))

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
