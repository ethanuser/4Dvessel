$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$PathsFile = "configs\paths.json"
if (-Not (Test-Path $PathsFile)) {
    Write-Host "Error: configs\paths.json not found." -ForegroundColor Red
    Write-Host "Please copy configs\paths.example.json to configs\paths.json and edit it." -ForegroundColor Yellow
    exit 1
}

$Paths = Get-Content $PathsFile -Raw | ConvertFrom-Json
$GaussiansPath = $Paths.4dgaussians_repo

if (-Not (Test-Path $GaussiansPath)) {
    Write-Host "Error: 4DGaussians repo not found at '$GaussiansPath'" -ForegroundColor Red
    exit 1
}

$GaussiansAbsPath = (Resolve-Path $GaussiansPath).Path

Write-Host "Running Stage 10: Export 4DGS Pointcloud" -ForegroundColor Cyan
Write-Host "Script: export_numpy_queue.py"
Write-Host "Environment: 4dgs"

Set-Location $GaussiansAbsPath
conda run -n 4dgs --no-capture-output python "export_numpy_queue.py" $args
Set-Location $RepoRoot
