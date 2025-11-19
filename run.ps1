param(
    [int]$FrontendPort = 8000,
    [int]$BackendPort = 5000,
    [string]$Step = "none",
    [string]$Config = "configs\\pipeline.yaml",
    [switch]$InstallDeps
)

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$backend = Join-Path $root "backend"
$frontend = Join-Path $root "web_app"
$venv = Join-Path $root ".venv"

$pythonCmd = "python"
if (-not (Get-Command $pythonCmd -ErrorAction SilentlyContinue)) { $pythonCmd = "py" }

if (-not (Test-Path $venv)) {
    & $pythonCmd -m venv $venv
}

$py = Join-Path $venv "Scripts\python.exe"
if ($InstallDeps) {
    & $py -m pip install -r (Join-Path $root "requirements.txt")
    if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed" }
}

$bestModel = Join-Path $root "models\best_model_torch.pth"
$finalModel = Join-Path $root "models\final_model_torch.pth"
if (-not (Test-Path $bestModel) -and -not (Test-Path $finalModel)) {
    Write-Host "Warning: trained model not found in models/. Run training before starting backend." -ForegroundColor Yellow
}

if ($Step -ne "none") {
    & $py (Join-Path $root "ml\pest_pipeline.py") --step $Step --config $Config
}

$backendProc = Start-Process -FilePath $py -ArgumentList (Join-Path $backend "app.py") -WorkingDirectory $backend -PassThru
$frontendProc = Start-Process -FilePath $py -ArgumentList ("-m http.server {0}" -f $FrontendPort) -WorkingDirectory $frontend -PassThru

Start-Process ("http://localhost:{0}" -f $FrontendPort)

Write-Host ("Backend started (PID {0}) on port {1}" -f $backendProc.Id, $BackendPort)
Write-Host ("Frontend started (PID {0}) on http://localhost:{1}" -f $frontendProc.Id, $FrontendPort)