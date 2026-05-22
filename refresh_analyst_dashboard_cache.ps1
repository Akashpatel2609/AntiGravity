$ErrorActionPreference = "Stop"

$Workspace = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = if ($env:DASHBOARD_PYTHON) { $env:DASHBOARD_PYTHON } else { "python" }
$Log = Join-Path $Workspace "analyst_dashboard_refresh.log"

Set-Location $Workspace
"[$(Get-Date -Format "yyyy-MM-dd HH:mm:ss")] Starting analyst dashboard refresh" | Add-Content -LiteralPath $Log

$script = @'
from analyst_dashboard_app import analyze

class DummyArgs:
    def __init__(self, ticker):
        self.ticker = ticker
    def get(self, key, default=None):
        return self.ticker if key == "ticker" else default

class DummyRequest:
    def __init__(self, ticker):
        self.args = DummyArgs(ticker)

import analyst_dashboard_app as appmod

for ticker in ["SPY", "QQQ", "NVDA", "MSFT", "AAPL", "AMD", "TSLA"]:
    with appmod.app.test_request_context(f"/api/analyze?ticker={ticker}"):
        response = appmod.analyze()
        print(ticker, response.status_code if hasattr(response, "status_code") else "ok")
'@

$output = $script | & $Python -
$output | ForEach-Object { $_.ToString() } | Add-Content -LiteralPath $Log
"[$(Get-Date -Format "yyyy-MM-dd HH:mm:ss")] Finished analyst dashboard refresh" | Add-Content -LiteralPath $Log
