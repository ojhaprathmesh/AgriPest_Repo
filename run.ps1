param(
    [int]$FrontendPort = 8000,
    [int]$BackendPort = 5000
)

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$backend = Join-Path $root "backend"
$frontend = Join-Path $root "dsfm"
$venv = Join-Path $backend ".venv"

$pythonCmd = "python"
if (-not (Get-Command $pythonCmd -ErrorAction SilentlyContinue)) { $pythonCmd = "py" }

if (-not (Test-Path (Join-Path $root "best_efficientnetb0.h5"))) {
    Write-Host "Warning: best_efficientnetb0.h5 not found in project root." -ForegroundColor Yellow
}

if (-not (Test-Path $venv)) {
    & $pythonCmd -m venv $venv
}

$backendPy = Join-Path $venv "Scripts\python.exe"
& $backendPy -m pip install -r (Join-Path $backend "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed" }

$backendProc = Start-Process -FilePath $backendPy -ArgumentList "app.py" -WorkingDirectory $backend -PassThru

$frontPython = $pythonCmd
if (-not (Get-Command $frontPython -ErrorAction SilentlyContinue)) { $frontPython = $backendPy }
$frontendProc = Start-Process -FilePath $frontPython -ArgumentList ("-m http.server {0}" -f $FrontendPort) -WorkingDirectory $frontend -PassThru

Start-Process ("http://localhost:{0}" -f $FrontendPort)

Write-Host ("Backend started (PID {0}) on port {1}" -f $backendProc.Id, $BackendPort)
Write-Host ("Frontend started (PID {0}) on http://localhost:{1}" -f $frontendProc.Id, $FrontendPort)