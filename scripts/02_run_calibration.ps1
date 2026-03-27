$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

Write-Host "Running Stage 3: Run Calibration" -ForegroundColor Cyan
Write-Host "Script: Preprocess\Camera Calibration\gen_calib_data_images.py"
Write-Host "Environment: sam2"

conda run -n sam2 --no-capture-output python "Preprocess\Camera Calibration\gen_calib_data_images.py" $args
