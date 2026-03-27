$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

Write-Host "Running Stage 8: Create NeRF Dataset" -ForegroundColor Cyan
Write-Host "Script: Preprocess\Create Nerf Datasets\create_nerf_data.py"
Write-Host "Environment: sam2"

conda run -n sam2 --no-capture-output python "Preprocess\Create Nerf Datasets\create_nerf_data.py" $args
