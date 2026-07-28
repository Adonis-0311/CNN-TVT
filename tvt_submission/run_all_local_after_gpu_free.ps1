param(
    [ValidateSet("preflight", "formal", "candidate-v4", "all")]
    [string]$Plan = "preflight",
    [switch]$Execute,
    [switch]$WriteRelease,
    [string]$Acknowledgement = "",
    [string]$Python = "D:\Python\python.exe",
    [int]$MinimumFreeGpuMiB = 7000
)

$ErrorActionPreference = "Stop"
$RequiredAcknowledgement = "START_TVT_ONLY_WHEN_MACHINE_IS_IDLE"
$PackageRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $PackageRoot
$FormalRunner = Join-Path $PackageRoot "run_local.ps1"
$CandidateRunner = Join-Path $PackageRoot "run_candidate_local.ps1"
$MacroGenerator = Join-Path $PackageRoot "generate_macro_values.py"
$FreezePath = Join-Path $PackageRoot "configs\formal_tvt_freeze_v1.json"

function Invoke-Checked {
    param(
        [string]$Program,
        [string[]]$Arguments,
        [string]$Failure
    )
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw $Failure
    }
}

function Get-ForeignComputeProcesses {
    $ProjectToken = [System.IO.Path]::GetFullPath($ProjectRoot)
    return @(
        Get-CimInstance Win32_Process |
            Where-Object {
                $_.Name -match "^(python|pythonw|matlab|MATLAB|soffice)(\.exe)?$" -and
                (
                    -not $_.CommandLine -or
                    $_.CommandLine.IndexOf(
                        $ProjectToken,
                        [System.StringComparison]::OrdinalIgnoreCase
                    ) -lt 0
                )
            } |
            Select-Object ProcessId, ParentProcessId, Name, CreationDate, CommandLine
    )
}

function Assert-MachineIdle {
    if ($MinimumFreeGpuMiB -le 0) {
        throw "MinimumFreeGpuMiB must be positive."
    }
    $Foreign = @(Get-ForeignComputeProcesses)
    if ($Foreign.Count -gt 0) {
        $Summary = $Foreign |
            ForEach-Object {
                "PID=$($_.ProcessId) Name=$($_.Name) Command=$($_.CommandLine)"
            }
        throw (
            "Foreign Python/MATLAB/LibreOffice work is active. " +
            "Nothing was stopped and TVT execution is refused.`n" +
            ($Summary -join "`n")
        )
    }

    $NvidiaSmi = Get-Command "nvidia-smi" -ErrorAction SilentlyContinue
    if (-not $NvidiaSmi) {
        throw "nvidia-smi is unavailable; GPU-idle state cannot be proven."
    }
    $FreeValues = @(
        & $NvidiaSmi.Source `
            --query-gpu=memory.free `
            --format=csv,noheader,nounits |
            ForEach-Object {
                $Value = 0
                if (-not [int]::TryParse($_.Trim(), [ref]$Value)) {
                    throw "Could not parse nvidia-smi free-memory output: $_"
                }
                $Value
            }
    )
    if ($LASTEXITCODE -ne 0 -or $FreeValues.Count -eq 0) {
        throw "GPU free-memory query failed; execution is refused."
    }
    $MinimumObserved = ($FreeValues | Measure-Object -Minimum).Minimum
    if ($MinimumObserved -lt $MinimumFreeGpuMiB) {
        throw (
            "Only $MinimumObserved MiB GPU memory is free; " +
            "$MinimumFreeGpuMiB MiB is required. No work was started."
        )
    }
    Write-Host "Idle gate passed: no foreign compute process; minimum free GPU memory is $MinimumObserved MiB."
}

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable not found: $Python"
}
foreach ($Required in @(
    $FormalRunner,
    $CandidateRunner,
    $MacroGenerator,
    $FreezePath
)) {
    if (-not (Test-Path -LiteralPath $Required -PathType Leaf)) {
        throw "Required local-pipeline file is missing: $Required"
    }
}
if ($WriteRelease -and -not $Execute) {
    throw "WriteRelease is invalid without Execute."
}
if ($Plan -eq "preflight" -and $Execute) {
    throw "The preflight plan never accepts Execute."
}

Set-Location -LiteralPath $ProjectRoot
Write-Host "TVT queued plan: $Plan"
Write-Host "Execution requested: $Execute"
Write-Host "This wrapper never kills, pauses, resumes, or signals another process."

if (-not $Execute) {
    & $FormalRunner -Python $Python
    if ($LASTEXITCODE -ne 0) {
        throw "Formal dry preflight failed."
    }
    & $CandidateRunner -Python $Python
    if ($LASTEXITCODE -ne 0) {
        throw "Candidate dry preflight failed."
    }
    Write-Host "Queue dry-run complete. No cache, simulation, training, candidate, macro, or release write was started."
    exit 0
}

if ($Acknowledgement -ne $RequiredAcknowledgement) {
    throw (
        "Execution acknowledgement mismatch. Pass " +
        "-Acknowledgement '$RequiredAcknowledgement' only after confirming " +
        "that the machine is idle."
    )
}
Assert-MachineIdle

$Freeze = Get-Content -LiteralPath $FreezePath -Raw | ConvertFrom-Json
$RunRoot = Join-Path $ProjectRoot $Freeze.experiment.expected_run_directory
$RunJson = Join-Path $RunRoot "run.json"
$MacroValues = Join-Path $RunRoot "formal_macro_values.json"

if ($Plan -in @("formal", "all")) {
    & $FormalRunner -Stage cache -Execute -Python $Python
    if ($LASTEXITCODE -ne 0) {
        throw "Formal cache stage failed."
    }
    & $FormalRunner -Stage experiment -Execute -Python $Python
    if ($LASTEXITCODE -ne 0) {
        throw "Formal experiment stage failed."
    }
    if (-not (Test-Path -LiteralPath $RunJson -PathType Leaf)) {
        throw "Formal run.json is absent after the experiment stage: $RunJson"
    }
    if (Test-Path -LiteralPath $MacroValues) {
        throw "Macro manifest already exists; refusing to overwrite: $MacroValues"
    }
    Invoke-Checked `
        -Program $Python `
        -Arguments @(
            "-B",
            $MacroGenerator,
            "--run-json", $RunJson,
            "--output", $MacroValues
        ) `
        -Failure "Artifact-derived macro/table generation failed."

    $MacroRelative = [System.IO.Path]::GetRelativePath(
        $ProjectRoot,
        $MacroValues
    )
    $ReleaseArguments = @{
        Stage = "release"
        MacroValues = $MacroRelative
        Execute = $true
        Python = $Python
    }
    if ($WriteRelease) {
        $ReleaseArguments["WriteRelease"] = $true
    }
    & $FormalRunner @ReleaseArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Formal release validation failed."
    }
    if (-not $WriteRelease) {
        Write-Host "Formal evidence preflight passed, but paper release files were not written."
        Write-Host "Re-run with -WriteRelease only after the human release review."
    }
}

if ($Plan -in @("candidate-v4", "all")) {
    & $CandidateRunner -Execute -Python $Python
    if ($LASTEXITCODE -ne 0) {
        throw "The one-shot VIMD-v4 candidate diagnostic failed."
    }
}

Write-Host "Requested queued plan completed without controlling any pre-existing process."

