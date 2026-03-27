$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

Write-Host "Running Stage 13: Cluster Mesh Export" -ForegroundColor Cyan
Write-Host "Script: vessel-stress-analysis\scripts\run_cluster_mesh_export.py"
Write-Host "Environment: stress"

conda run -n stress --no-capture-output python "vessel-stress-analysis\scripts\run_cluster_mesh_export.py" $args
