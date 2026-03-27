$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

Write-Host "Running Stage 7: Merge Masked Videos" -ForegroundColor Cyan
Write-Host "Script: sam2\merge_videos.py"
Write-Host "Environment: sam2"

conda run -n sam2 --no-capture-output python "sam2\merge_videos.py" $args
