$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

Write-Host "Running Stage 14: Stress Analysis" -ForegroundColor Cyan
Write-Host "Script: vessel-stress-analysis\scripts\run_stress_analysis.py"
Write-Host "Environment: stress"

conda run -n stress --no-capture-output python "vessel-stress-analysis\scripts\run_stress_analysis.py" $args
