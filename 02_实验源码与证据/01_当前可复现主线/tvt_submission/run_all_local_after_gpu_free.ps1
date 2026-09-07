param(
    [ValidateSet("preflight", "formal", "release", "candidate-v4", "all")]
    [string]$Plan = "preflight",
    [switch]$Execute,
    [switch]$WriteRelease,
    [switch]$AllowValidatedReuse,
    [string]$ExpectedCacheManifestSha256 = "",
    [string]$ExpectedRunJsonSha256 = "",
    [string]$Acknowledgement = "",
    [string]$Python = "D:\Python\python.exe",
    [int]$MinimumFreeGpuMiB = 7000,
    [int]$MinimumFreeDiskGiB = 20
)

$ErrorActionPreference = "Stop"
$RequiredAcknowledgement = "START_TVT_ONLY_WHEN_MACHINE_IS_IDLE"
$ExecutionMutexName = "Global\VIMD_AMC_TVT_LOCAL_EXECUTION_V1"
$PackageRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $PackageRoot
$FormalRunner = Join-Path $PackageRoot "run_local.ps1"
$CandidateRunner = Join-Path $PackageRoot "run_candidate_local.ps1"
$MacroGenerator = Join-Path $PackageRoot "generate_macro_values.py"
$FreezePath = Join-Path $PackageRoot "configs\formal_tvt_freeze_v1.json"
$MacroValues = Join-Path $PackageRoot "formal_macro_values.json"
$ExecutionMutex = $null

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

function Get-ActiveComputeProcesses {
    return @(
        Get-CimInstance Win32_Process |
            Where-Object {
                $_.Name -match (
                    "^(python|pythonw|matlab|matlabworker|soffice)" +
                    "(\.exe|\.bin)?$"
                )
            } |
            Select-Object ProcessId, ParentProcessId, Name, CreationDate, CommandLine
    )
}

function Assert-ExecutionAcknowledged {
    if ($Acknowledgement -ne $RequiredAcknowledgement) {
        throw (
            "Execution acknowledgement mismatch. Pass " +
            "-Acknowledgement '$RequiredAcknowledgement' only after confirming " +
            "that the machine is idle."
        )
    }
}

function Enter-ExecutionMutex {
    $CreatedNew = $false
    try {
        $script:ExecutionMutex = [System.Threading.Mutex]::new(
            $false,
            $ExecutionMutexName,
            [ref]$CreatedNew
        )
        if (-not $script:ExecutionMutex.WaitOne(0)) {
            throw (
                "Another TVT local execution owns mutex " +
                "'$ExecutionMutexName'. Nothing was started."
            )
        }
    } catch [System.Threading.AbandonedMutexException] {
        if ($script:ExecutionMutex) {
            try {
                $script:ExecutionMutex.ReleaseMutex()
            } catch {
                # The abandoned handle is disposed below; never continue execution.
            }
            $script:ExecutionMutex.Dispose()
            $script:ExecutionMutex = $null
        }
        throw (
            "The TVT execution mutex was abandoned by an earlier process. " +
            "Execution is refused until its outputs are audited and the command " +
            "is run again."
        )
    } catch {
        if ($script:ExecutionMutex) {
            $script:ExecutionMutex.Dispose()
            $script:ExecutionMutex = $null
        }
        throw
    }
}

function Exit-ExecutionMutex {
    if ($script:ExecutionMutex) {
        try {
            $script:ExecutionMutex.ReleaseMutex()
        } finally {
            $script:ExecutionMutex.Dispose()
            $script:ExecutionMutex = $null
        }
    }
}

function Assert-MachineIdle {
    if ($MinimumFreeGpuMiB -le 0) {
        throw "MinimumFreeGpuMiB must be positive."
    }
    if ($MinimumFreeDiskGiB -le 0) {
        throw "MinimumFreeDiskGiB must be positive."
    }
    $Active = @(Get-ActiveComputeProcesses)
    if ($Active.Count -gt 0) {
        $Summary = $Active |
            ForEach-Object {
                "PID=$($_.ProcessId) Name=$($_.Name) Command=$($_.CommandLine)"
            }
        throw (
            "Python/MATLAB/LibreOffice work is active, including work under " +
            "this project. " +
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

    $DriveRoot = [System.IO.Path]::GetPathRoot(
        [System.IO.Path]::GetFullPath($ProjectRoot)
    )
    $FreeDiskBytes = [System.IO.DriveInfo]::new(
        $DriveRoot
    ).AvailableFreeSpace
    $RequiredDiskBytes = [int64]$MinimumFreeDiskGiB * 1GB
    if ($FreeDiskBytes -lt $RequiredDiskBytes) {
        $FreeDiskGiB = [math]::Round($FreeDiskBytes / 1GB, 2)
        throw (
            "Only $FreeDiskGiB GiB is free on $DriveRoot; " +
            "$MinimumFreeDiskGiB GiB is required. No work was started."
        )
    }
    $FreeDiskGiB = [math]::Round($FreeDiskBytes / 1GB, 2)
    Write-Host (
        "Idle gate passed: no active Python/MATLAB/LibreOffice process; " +
        "minimum free GPU memory is $MinimumObserved MiB; " +
        "free project-volume space is $FreeDiskGiB GiB."
    )
}

function Assert-Sha256Syntax {
    param(
        [string]$Value,
        [string]$Name
    )
    if ($Value -notmatch "^[0-9a-fA-F]{64}$") {
        throw "$Name must be an exact 64-hex SHA-256 digest."
    }
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
if ($WriteRelease -and $Plan -notin @("formal", "release", "all")) {
    throw "WriteRelease is valid only for the formal, release, or all plan."
}
if (
    ($ExpectedCacheManifestSha256 -or $ExpectedRunJsonSha256) -and
    -not $AllowValidatedReuse
) {
    throw "Expected reuse digests require -AllowValidatedReuse."
}
if ($ExpectedCacheManifestSha256) {
    Assert-Sha256Syntax `
        -Value $ExpectedCacheManifestSha256 `
        -Name "ExpectedCacheManifestSha256"
}
if ($ExpectedRunJsonSha256) {
    Assert-Sha256Syntax `
        -Value $ExpectedRunJsonSha256 `
        -Name "ExpectedRunJsonSha256"
}
if ($Plan -eq "preflight" -and $Execute) {
    throw "The preflight plan never accepts Execute."
}

Set-Location -LiteralPath $ProjectRoot
Write-Host "TVT queued plan: $Plan"
Write-Host "Execution requested: $Execute"
Write-Host "This wrapper never kills, pauses, resumes, or signals another process."

if (-not $Execute) {
    if ($Plan -in @("preflight", "formal", "all")) {
        & $FormalRunner -Python $Python
        if ($LASTEXITCODE -ne 0) {
            throw "Formal dry preflight failed."
        }
    }
    if ($Plan -eq "release") {
        $Freeze = Get-Content -LiteralPath $FreezePath -Raw | ConvertFrom-Json
        $DryRunRoot = Join-Path (
            $ProjectRoot
        ) $Freeze.experiment.expected_run_directory
        $DryRunJson = Join-Path $DryRunRoot "run.json"
        $MacroRelative = [System.IO.Path]::GetRelativePath(
            $ProjectRoot,
            $MacroValues
        )
        & $FormalRunner `
            -Stage release `
            -MacroValues $MacroRelative `
            -Python $Python
        if ($LASTEXITCODE -ne 0) {
            throw "Release dry preflight failed."
        }
        if (-not (Test-Path -LiteralPath $DryRunJson -PathType Leaf)) {
            Write-Host "Release remains gated because formal run.json is absent."
        }
    }
    if ($Plan -in @("preflight", "candidate-v4", "all")) {
        & $CandidateRunner -Python $Python
        if ($LASTEXITCODE -ne 0) {
            throw "Candidate dry preflight failed."
        }
    }
    Write-Host "Queue dry-run complete. No cache, simulation, training, candidate, macro, or release write was started."
    exit 0
}

Assert-ExecutionAcknowledged
Enter-ExecutionMutex
try {
    Assert-MachineIdle

    if ($Plan -in @("formal", "release", "all")) {
        $Freeze = Get-Content -LiteralPath $FreezePath -Raw | ConvertFrom-Json
        $CacheRoot = Join-Path $ProjectRoot $Freeze.cache.output
        $CacheManifest = Join-Path $CacheRoot "manifest.json"
        $RunRoot = Join-Path (
            $ProjectRoot
        ) $Freeze.experiment.expected_run_directory
        $RunJson = Join-Path $RunRoot "run.json"
    }

    if ($Plan -in @("formal", "all")) {
        $CacheExistedAtStart = Test-Path -LiteralPath $CacheRoot
        $RunExistedAtStart = Test-Path -LiteralPath $RunRoot
        if ($CacheExistedAtStart -and -not $AllowValidatedReuse) {
            throw (
                "Formal cache already exists. Default execution refuses reuse: " +
                "$CacheRoot"
            )
        }
        if ($RunExistedAtStart -and -not $AllowValidatedReuse) {
            throw (
                "Formal run directory already exists. Default execution refuses " +
                "reuse: $RunRoot"
            )
        }

        $CacheStageArguments = @{
            Stage = "cache"
            Execute = $true
            Acknowledgement = $Acknowledgement
            MinimumFreeGpuMiB = $MinimumFreeGpuMiB
            MinimumFreeDiskGiB = $MinimumFreeDiskGiB
            Python = $Python
        }
        if ($AllowValidatedReuse) {
            $CacheStageArguments["AllowValidatedReuse"] = $true
            $CacheStageArguments[
                "ExpectedCacheManifestSha256"
            ] = $ExpectedCacheManifestSha256
        }
        & $FormalRunner @CacheStageArguments
        if ($LASTEXITCODE -ne 0) {
            throw "Formal cache stage failed."
        }
        if (-not (Test-Path -LiteralPath $CacheManifest -PathType Leaf)) {
            throw "Formal cache manifest is absent after cache stage: $CacheManifest"
        }
        $PinnedCacheDigest = (
            Get-FileHash -Algorithm SHA256 -LiteralPath $CacheManifest
        ).Hash.ToLowerInvariant()

        Assert-MachineIdle
        $ExperimentStageArguments = @{
            Stage = "experiment"
            Execute = $true
            AllowValidatedReuse = $true
            ExpectedCacheManifestSha256 = $PinnedCacheDigest
            Acknowledgement = $Acknowledgement
            MinimumFreeGpuMiB = $MinimumFreeGpuMiB
            MinimumFreeDiskGiB = $MinimumFreeDiskGiB
            Python = $Python
        }
        if ($RunExistedAtStart) {
            $ExperimentStageArguments[
                "ExpectedRunJsonSha256"
            ] = $ExpectedRunJsonSha256
        }
        & $FormalRunner @ExperimentStageArguments
        if ($LASTEXITCODE -ne 0) {
            throw "Formal experiment stage failed."
        }
        if (-not (Test-Path -LiteralPath $RunJson -PathType Leaf)) {
            throw "Formal run.json is absent after experiment stage: $RunJson"
        }
        $PinnedRunDigest = (
            Get-FileHash -Algorithm SHA256 -LiteralPath $RunJson
        ).Hash.ToLowerInvariant()

        if (Test-Path -LiteralPath $MacroValues) {
            throw "Macro manifest already exists; refusing to overwrite: $MacroValues"
        }
        Assert-MachineIdle
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
            AllowValidatedReuse = $true
            ExpectedCacheManifestSha256 = $PinnedCacheDigest
            ExpectedRunJsonSha256 = $PinnedRunDigest
            Acknowledgement = $Acknowledgement
            MinimumFreeGpuMiB = $MinimumFreeGpuMiB
            MinimumFreeDiskGiB = $MinimumFreeDiskGiB
            Python = $Python
        }
        if ($WriteRelease) {
            $ReleaseArguments["WriteRelease"] = $true
        }
        Assert-MachineIdle
        & $FormalRunner @ReleaseArguments
        if ($LASTEXITCODE -ne 0) {
            throw "Formal release validation failed."
        }
        if (-not $WriteRelease) {
            Write-Host "Formal evidence preflight passed, but paper release files were not written."
            Write-Host "Re-run with -WriteRelease only after the human release review."
        }
    }

    if ($Plan -eq "release") {
        if (-not $AllowValidatedReuse) {
            throw (
                "The release plan consumes an existing formal cache and run. " +
                "Pass -AllowValidatedReuse with their pinned SHA-256 digests."
            )
        }
        if (-not (Test-Path -LiteralPath $MacroValues -PathType Leaf)) {
            throw "Audited macro manifest is absent: $MacroValues"
        }
        $MacroRelative = [System.IO.Path]::GetRelativePath(
            $ProjectRoot,
            $MacroValues
        )
        $ReleaseArguments = @{
            Stage = "release"
            MacroValues = $MacroRelative
            Execute = $true
            AllowValidatedReuse = $true
            ExpectedCacheManifestSha256 = $ExpectedCacheManifestSha256
            ExpectedRunJsonSha256 = $ExpectedRunJsonSha256
            Acknowledgement = $Acknowledgement
            MinimumFreeGpuMiB = $MinimumFreeGpuMiB
            MinimumFreeDiskGiB = $MinimumFreeDiskGiB
            Python = $Python
        }
        if ($WriteRelease) {
            $ReleaseArguments["WriteRelease"] = $true
        }
        Assert-MachineIdle
        & $FormalRunner @ReleaseArguments
        if ($LASTEXITCODE -ne 0) {
            throw "Formal release validation failed."
        }
    }

    if ($Plan -in @("candidate-v4", "all")) {
        Assert-MachineIdle
        & $CandidateRunner `
            -Execute `
            -Acknowledgement $Acknowledgement `
            -MinimumFreeGpuMiB $MinimumFreeGpuMiB `
            -MinimumFreeDiskGiB $MinimumFreeDiskGiB `
            -Python $Python
        if ($LASTEXITCODE -ne 0) {
            throw "The one-shot VIMD-v4 candidate diagnostic failed."
        }
    }
} finally {
    Exit-ExecutionMutex
}

Write-Host "Requested queued plan completed without controlling any pre-existing process."
