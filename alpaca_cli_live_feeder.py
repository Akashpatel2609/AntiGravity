from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ENV_FILE = ROOT / "alpaca_credentials.env"
CACHE_FILE = ROOT / "alpaca_live_cache.json"
LOG_FILE = ROOT / "alpaca_cli_live_feeder.log"
ALPACA_EXE = Path(os.getenv("ALPACA_CLI_PATH", "alpaca"))
PROFILE = "codex-dashboard"
DEFAULT_SYMBOLS = ["SPY", "QQQ", "NVDA", "MSFT", "AAPL", "AMD", "TSLA"]


def load_env() -> dict[str, str]:
    env = os.environ.copy()
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip().strip('"')
    return env


def log(message: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    LOG_FILE.open("a", encoding="utf-8").write(f"[{stamp}] {message}\n")


def run_cli(args: list[str], env: dict[str, str]) -> dict:
    if not (ALPACA_EXE.exists() or shutil.which(str(ALPACA_EXE))):
        raise RuntimeError("alpaca CLI not found. Install it or set ALPACA_CLI_PATH.")
    proc = subprocess.run(
        [str(ALPACA_EXE), *args],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout).strip())
    return json.loads(proc.stdout)


def fetch_symbol(symbol: str, env: dict[str, str]) -> dict:
    trade = run_cli(["data", "latest-trade", "--symbol", symbol, "--feed", "iex", "--profile", PROFILE, "--quiet"], env)
    quote = run_cli(["data", "latest-quote", "--symbol", symbol, "--feed", "iex", "--profile", PROFILE, "--quiet"], env)
    trade_data = trade.get("trade", {})
    quote_data = quote.get("quote", {})
    bid = quote_data.get("bp")
    ask = quote_data.get("ap")
    mid = None
    if isinstance(bid, (int, float)) and isinstance(ask, (int, float)) and bid > 0 and ask > 0:
        mid = (bid + ask) / 2
    return {
        "symbol": symbol,
        "source": "Alpaca CLI latest trade + quote",
        "feed": "IEX/free",
        "last": trade_data.get("p"),
        "lastSize": trade_data.get("s"),
        "lastExchange": trade_data.get("x"),
        "lastTimestamp": trade_data.get("t"),
        "bid": bid,
        "ask": ask,
        "mid": mid,
        "bidSize": quote_data.get("bs"),
        "askSize": quote_data.get("as"),
        "quoteTimestamp": quote_data.get("t"),
        "updated": datetime.now(timezone.utc).isoformat(),
    }


def write_cache(cache: dict) -> None:
    temp = CACHE_FILE.with_suffix(".tmp")
    temp.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    temp.replace(CACHE_FILE)


def main() -> None:
    env = load_env()
    symbols = os.getenv("ALPACA_FEED_SYMBOLS")
    watchlist = [s.strip().upper() for s in symbols.split(",")] if symbols else DEFAULT_SYMBOLS
    interval = int(os.getenv("ALPACA_FEED_INTERVAL_SECONDS", "20"))
    log(f"Starting Alpaca CLI feeder for {','.join(watchlist)} every {interval}s")
    cache: dict[str, dict] = {}
    while True:
        for symbol in watchlist:
            try:
                cache[symbol] = fetch_symbol(symbol, env)
            except Exception as exc:
                cache[symbol] = {
                    "symbol": symbol,
                    "source": "Alpaca CLI",
                    "error": str(exc),
                    "updated": datetime.now(timezone.utc).isoformat(),
                }
                log(f"{symbol}: {exc}")
        write_cache({"updated": datetime.now(timezone.utc).isoformat(), "symbols": cache})
        time.sleep(interval)


if __name__ == "__main__":
    main()
