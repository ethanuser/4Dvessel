$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

Write-Host "Running Stage 2: Extract Calibration Images" -ForegroundColor Cyan
Write-Host "Script: Preprocess\Camera Calibration\create_calibration_images.py"
Write-Host "Environment: sam2"

conda run -n sam2 --no-capture-output python "Preprocess\Camera Calibration\create_calibration_images.py" $args
