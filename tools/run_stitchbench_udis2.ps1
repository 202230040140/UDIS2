param(
    [string]$Manifest = "D:\StitchBench_Result\manifest.csv",
    [string]$OutRoot = "D:\StitchBench_Result\_global_work\udis2",
    [string]$Python = "C:\Users\22499\.venvs\obj-gsp-sam\Scripts\python.exe",
    [string]$Gpu = "0",
    [int]$MaxSide = 768,
    [int]$MaxIter = 50,
    [int]$Limit = 0,
    [string[]]$Scene,
    [switch]$SkipExisting,
    [switch]$Force
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$args = @(
    "reproduce\run_stitchbench.py",
    "--manifest", $Manifest,
    "--out-root", $OutRoot,
    "--gpu", $Gpu,
    "--python", $Python,
    "--max-side", $MaxSide,
    "--max-iter", $MaxIter
)
if ($Limit -gt 0) { $args += @("--limit", $Limit) }
if ($Scene) { foreach ($s in $Scene) { $args += @("--scene", $s) } }
if ($SkipExisting) { $args += "--skip-existing" }
if ($Force) { $args += "--force" }
$env:PYTHONUNBUFFERED = "1"
& $Python @args
if ($LASTEXITCODE -ne 0) { throw "run_stitchbench.py failed with exit code $LASTEXITCODE" }
