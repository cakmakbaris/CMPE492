$project = Split-Path -Parent $PSScriptRoot
$python = Join-Path $project '.venv/Scripts/python.exe'
Push-Location $project
try {
    & $python -m src.main @args
    $status = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $status
