$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

Write-Host "Running Stage 0b: Downsample Videos" -ForegroundColor Cyan
Write-Host "Script: Preprocess\Filter Background\downsample_videos.py"
Write-Host "Environment: sam2"

conda run -n sam2 --no-capture-output python "Preprocess\Filter Background\downsample_videos.py" $args
