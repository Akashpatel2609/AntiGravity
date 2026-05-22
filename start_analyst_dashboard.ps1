$ErrorActionPreference = "Stop"

$Workspace = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = if ($env:DASHBOARD_PYTHON) { $env:DASHBOARD_PYTHON } else { "python" }
$App = Join-Path $Workspace "analyst_dashboard_app.py"
$Log = Join-Path $Workspace "analyst_dashboard_server.log"
$ErrLog = Join-Path $Workspace "analyst_dashboard_server_error.log"
$StatusLog = Join-Path $Workspace "analyst_dashboard_status.log"
$Port = 8766

$connections = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue
$processIds = $connections | Select-Object -ExpandProperty OwningProcess -Unique | Where-Object { $_ -and $_ -ne 0 }
foreach ($processId in $processIds) {
    try {
        $proc = Get-Process -Id $processId -ErrorAction Stop
        if ($proc.ProcessName -match "python") {
            Stop-Process -Id $processId -Force
        }
    } catch {
    }
}

Set-Location $Workspace
"[$(Get-Date -Format "yyyy-MM-dd HH:mm:ss")] Starting analyst dashboard on http://127.0.0.1:$Port" | Add-Content -LiteralPath $StatusLog
Start-Process -FilePath $Python -ArgumentList @($App) -WorkingDirectory $Workspace -WindowStyle Hidden -RedirectStandardOutput $Log -RedirectStandardError $ErrLog
Start-Sleep -Seconds 3
try {
    $status = (Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:$Port/" -TimeoutSec 10).StatusCode
    "[$(Get-Date -Format "yyyy-MM-dd HH:mm:ss")] Analyst dashboard status $status" | Add-Content -LiteralPath $StatusLog
    "http://127.0.0.1:$Port/"
} catch {
    "[$(Get-Date -Format "yyyy-MM-dd HH:mm:ss")] Analyst dashboard failed: $($_.Exception.Message)" | Add-Content -LiteralPath $StatusLog
    throw
}
