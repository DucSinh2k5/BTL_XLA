param([int]$Port = 8000)
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$venvPython = Join-Path (Split-Path $projectRoot -Parent) '.venv\Scripts\python.exe'
if (Test-Path -LiteralPath $venvPython) {
    $webPython = $venvPython
} else {
    $webPython = (Get-Command python -ErrorAction Stop).Source
}
Write-Host "SUN397 demo: http://127.0.0.1:$Port/ and http://127.0.0.1:$Port/mtacs"
Write-Host 'TACS, M-TACS, Florence and CLIP preload automatically. Wait for the Ready badge.'
& $webPython -m uvicorn --app-dir (Join-Path $projectRoot 'apps') sun397_demo.app:app --host 127.0.0.1 --port $Port
exit $LASTEXITCODE
