# AntiGravity

Local stock research dashboard built with Flask, Plotly, yfinance, Alpaca live data, a transcript-inspired market-cycle model, and an observable Markov regime model.

## Hosted dashboard

Use Streamlit Community Cloud for the free hosted version, especially while the repository is private.

1. Go to [share.streamlit.io](https://share.streamlit.io/).
2. Connect GitHub and choose `Akashpatel2609/AntiGravity`.
3. Set the main file path to `streamlit_app.py`.
4. Add these app secrets if you want Alpaca live/after-hours data:

```toml
ALPACA_API_KEY = "your_key"
ALPACA_SECRET_KEY = "your_secret"
ALPACA_ENDPOINT = "https://paper-api.alpaca.markets/v2"
```

After deployment, put the Streamlit URL here:

```text
Live dashboard: (https://trade-researcher.streamlit.app/)
```

## Pages

- `/` - ticker research dashboard with live price, sentiment, macro, technicals, and six-stage bull/bear cycle scoring.
- `/markov-skill` - Markov hedge fund method: transition matrix, stationary distribution, n-step forecast, walk-forward Sharpe/max drawdown, optional HMM.
- `/fusion` - final verdict dashboard blending the transcript-cycle model with Markov regimes, macro/sentiment, and shock-event robustness.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item alpaca_credentials.env.example alpaca_credentials.env
```

Fill `alpaca_credentials.env` if you want Alpaca live/after-hours data. The app falls back to yfinance when Alpaca is unavailable.

## Run

```powershell
.\start_analyst_dashboard.ps1
```

Open:

```text
http://127.0.0.1:8766/
```

## Run with Streamlit

```powershell
streamlit run streamlit_app.py
```

Optional Alpaca CLI feeder:

```powershell
.\start_alpaca_cli_live_feeder.ps1
```

The feeder polls Alpaca CLI latest trade/quote and writes `alpaca_live_cache.json`, which is ignored by git.

## Notes

This is a research dashboard, not financial advice. Backtests are historical, and Markov transition matrices measure past regime behavior rather than guaranteeing future outcomes.
