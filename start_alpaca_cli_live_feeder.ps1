$ErrorActionPreference = "Stop"

$Workspace = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = if ($env:DASHBOARD_PYTHON) { $env:DASHBOARD_PYTHON } else { "python" }
$Feeder = Join-Path $Workspace "alpaca_cli_live_feeder.py"
$Log = Join-Path $Workspace "alpaca_cli_live_feeder_stdout.log"
$ErrLog = Join-Path $Workspace "alpaca_cli_live_feeder_stderr.log"

$existing = Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -and $_.CommandLine.Contains("alpaca_cli_live_feeder.py")
}
foreach ($proc in $existing) {
    Stop-Process -Id $proc.ProcessId -Force -ErrorAction SilentlyContinue
}

Set-Location $Workspace
Start-Process -FilePath $Python -ArgumentList @($Feeder) -WorkingDirectory $Workspace -WindowStyle Hidden -RedirectStandardOutput $Log -RedirectStandardError $ErrLog
Start-Sleep -Seconds 4

$cache = Join-Path $Workspace "alpaca_live_cache.json"
if (Test-Path -LiteralPath $cache) {
    "Alpaca CLI live feeder started. Cache: $cache"
} else {
    "Alpaca CLI live feeder started; cache is still warming."
}
