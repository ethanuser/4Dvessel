$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

Write-Host "Running Stage 6: SAM2 Masking GUI" -ForegroundColor Cyan
Write-Host "Script: sam2\mask_video_fastest.py"
Write-Host "Environment: sam2"

conda run -n sam2 --no-capture-output python "sam2\mask_video_fastest.py" $args
