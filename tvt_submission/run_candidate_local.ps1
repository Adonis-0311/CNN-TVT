param(
    [switch]$Execute,
    [string]$Python = "D:\Python\python.exe"
)

$ErrorActionPreference = "Stop"
$PackageRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $PackageRoot
$Script = "diagnostics\run_iqformer_route_v4_diagnostic.py"
$Arguments = @(
    $Script,
    "--cache-root", "standards\cache_factor_screening_1024_v1",
    "--output", "artifacts\diagnostic_iqformer_route_v4_dsbn",
    "--cpu-threads", "1",
    "--execute-preregistered-diagnostic"
)

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable not found: $Python"
}

Set-Location -LiteralPath $ProjectRoot
$Display = "$Python " + ($Arguments -join " ")
Write-Host $Display

if (-not $Execute) {
    Write-Host "Dry preflight complete. VIMD-v4 training was not started."
    Write-Host "The diagnostic is one-shot and non-evidence. Read docs\VIMD_V4_DSBN_PREREGISTRATION.md before using -Execute."
    exit 0
}

if (Test-Path -LiteralPath "artifacts\diagnostic_iqformer_route_v4_dsbn") {
    throw "Candidate output already exists; refusing to merge, overwrite, or tune after outcome."
}

& $Python @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "The preregistered VIMD-v4 diagnostic failed operationally."
}

Write-Host "Diagnostic finished. Its output remains non-evidence and must be judged only by the frozen all-required gate."

