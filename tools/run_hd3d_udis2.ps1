param(
    [string]$Manifest = "D:\HD3D_Result\_work\manifest.csv",
    [string]$ResultRoot = "D:\HD3D_Result",
    [string]$WorkRoot = "D:\HD3D_Result\_work",
    [string]$Python = "C:\Users\22499\.venvs\obj-gsp-sam\Scripts\python.exe",
    [string]$Gpu = "0",
    [string]$Device = "cuda",
    [int]$MaxSide = 768,
    [int]$MaxIter = 50,
    [int]$Limit = 0,
    [string[]]$Scene,
    [string[]]$Pair,
    [switch]$SkipExisting,
    [switch]$Force
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$args = @(
    "reproduce\run_hd3d.py",
    "--manifest", $Manifest,
    "--result-root", $ResultRoot,
    "--work-root", $WorkRoot,
    "--gpu", $Gpu,
    "--python", $Python,
    "--device", $Device,
    "--max-side", $MaxSide,
    "--max-iter", $MaxIter
)
if ($Limit -gt 0) { $args += @("--limit", $Limit) }
if ($Scene) { foreach ($s in $Scene) { $args += @("--scene", $s) } }
if ($Pair) { foreach ($p in $Pair) { $args += @("--pair", $p) } }
if ($SkipExisting) { $args += "--skip-existing" }
if ($Force) { $args += "--force" }
$env:PYTHONUNBUFFERED = "1"
& $Python @args
if ($LASTEXITCODE -ne 0) { throw "run_hd3d.py failed with exit code $LASTEXITCODE" }
