param(
    [switch]$Execute,
    [ValidateSet("all", "cache", "experiment", "release")]
    [string]$Stage = "all",
    [string]$MacroValues = "",
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
$ConfigPath = Join-Path $PackageRoot "configs\formal_tvt_freeze_v1.json"
$ValidatorPath = Join-Path $PackageRoot "validate_formal_freeze.py"
$ExecutionMutex = $null

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable not found: $Python"
}
if (-not (Test-Path -LiteralPath $ValidatorPath -PathType Leaf)) {
    throw "Formal-freeze validator not found: $ValidatorPath"
}
if ($Execute -and $Acknowledgement -ne $RequiredAcknowledgement) {
    throw (
        "Execution acknowledgement mismatch. Pass " +
        "-Acknowledgement '$RequiredAcknowledgement' only after confirming " +
        "that the machine is idle."
    )
}

& $Python -B $ValidatorPath --config $ConfigPath
if ($LASTEXITCODE -ne 0) {
    throw "Formal-freeze validation failed; no stage command was printed or executed."
}

$Config = Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json

$CacheRoot = Join-Path $ProjectRoot $Config.cache.output
$OutputBase = Join-Path $ProjectRoot $Config.experiment.output
$RunRoot = Join-Path $OutputBase $Config.experiment.run_id
$ExpectedRunRoot = Join-Path $ProjectRoot $Config.experiment.expected_run_directory
if ([System.IO.Path]::GetFullPath($RunRoot) -ne [System.IO.Path]::GetFullPath($ExpectedRunRoot)) {
    throw "Frozen output/run-id pair disagrees with expected_run_directory."
}
$Models = $Config.experiment.models -join ","
$Seeds = ($Config.experiment.seeds | ForEach-Object { "$_" }) -join ","
$Holm = $Config.experiment.holm_candidates -join ","
$CacheManifest = Join-Path $CacheRoot "manifest.json"
$RunJson = Join-Path $RunRoot "run.json"

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

function Assert-PinnedFileDigest {
    param(
        [string]$Path,
        [string]$ExpectedSha256,
        [string]$Label
    )
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Label is absent: $Path"
    }
    if ((Get-Item -LiteralPath $Path).Length -le 0) {
        throw "$Label is empty: $Path"
    }
    Assert-Sha256Syntax -Value $ExpectedSha256 -Name "expected $Label SHA-256"
    $Actual = (
        Get-FileHash -Algorithm SHA256 -LiteralPath $Path
    ).Hash.ToLowerInvariant()
    if ($Actual -ne $ExpectedSha256.ToLowerInvariant()) {
        throw (
            "$Label SHA-256 mismatch. Expected " +
            "$($ExpectedSha256.ToLowerInvariant()), observed $Actual."
        )
    }
    return $Actual
}

function Assert-CacheReuse {
    if (-not $AllowValidatedReuse) {
        throw (
            "Formal cache reuse is disabled by default. Pass " +
            "-AllowValidatedReuse with -ExpectedCacheManifestSha256 only " +
            "after a read-only audit."
        )
    }
    $Digest = Assert-PinnedFileDigest `
        -Path $CacheManifest `
        -ExpectedSha256 $ExpectedCacheManifestSha256 `
        -Label "formal cache manifest"
    Write-Host "Pinned formal cache manifest accepted for read-only reuse: $Digest"
    return $Digest
}

function Assert-RunReuse {
    if (-not $AllowValidatedReuse) {
        throw (
            "Formal run reuse is disabled by default. Pass " +
            "-AllowValidatedReuse with -ExpectedRunJsonSha256 only after " +
            "a read-only audit."
        )
    }
    $Digest = Assert-PinnedFileDigest `
        -Path $RunJson `
        -ExpectedSha256 $ExpectedRunJsonSha256 `
        -Label "formal run.json"
    Assert-RunRecordEligible
    Write-Host "Pinned eligible formal run accepted for read-only reuse: $Digest"
    return $Digest
}

function Assert-RunRecordEligible {
    $Record = Get-Content -LiteralPath $RunJson -Raw | ConvertFrom-Json
    if ($Record.run_id -ne $Config.experiment.run_id) {
        throw "Formal run.json has the wrong run_id."
    }
    if (
        $Record.status -ne "complete" -or
        $Record.execution_status -ne "complete"
    ) {
        throw "Formal run.json is not complete."
    }
    $Eligibility = $Record.evidence_eligibility
    if (
        $Eligibility.eligible -ne $true -or
        $Eligibility.formal_paper_evidence_eligible -ne $true -or
        $Eligibility.headline_eligible -ne $true
    ) {
        throw "Formal run.json is not eligible formal headline evidence."
    }
    $ExpectedModels = $Config.experiment.models -join "`n"
    $ObservedModels = @($Record.models) -join "`n"
    $ExpectedSeeds = $Config.experiment.seeds -join "`n"
    $ObservedSeeds = @($Record.seeds) -join "`n"
    if (
        $ObservedModels -ne $ExpectedModels -or
        $ObservedSeeds -ne $ExpectedSeeds
    ) {
        throw "Formal run.json model/seed matrix disagrees with the freeze."
    }
    if (
        $Record.comparison_protocol.reference_model -ne
        $Config.experiment.reference_model
    ) {
        throw "Formal run.json reference model disagrees with the freeze."
    }
}

$CacheArguments = @(
    "standards\build_factor_cache.py",
    "--output", $Config.cache.output,
    "--preset", $Config.cache.preset,
    "--sample-length", "$($Config.cache.sample_length)",
    "--guard-samples", "$($Config.cache.guard_samples)",
    "--master-seed", "$($Config.cache.master_seed)",
    "--matlab-timeout-s", "$($Config.cache.matlab_timeout_s)"
)

$ExperimentArguments = @(
    "experiments\run_standard_experiment.py",
    "--cache-root", $Config.cache.output,
    "--models", $Models,
    "--seeds", $Seeds,
    "--reference-model", $Config.experiment.reference_model,
    "--holm-candidates", $Holm,
    "--device", $Config.experiment.device,
    "--output", $Config.experiment.output,
    "--run-id", $Config.experiment.run_id,
    "--verify-checksums",
    "--validate-components",
    "--epochs", "$($Config.experiment.training.epochs)",
    "--batch-size", "$($Config.experiment.training.batch_size)",
    "--learning-rate", "$($Config.experiment.training.learning_rate)",
    "--weight-decay", "$($Config.experiment.training.weight_decay)",
    "--mask-start-epoch", "$($Config.experiment.training.mask_start_epoch)",
    "--contrastive-start-epoch", "$($Config.experiment.training.contrastive_start_epoch)",
    "--mask-ramp-epochs", "$($Config.experiment.training.mask_ramp_epochs)",
    "--contrastive-ramp-epochs", "$($Config.experiment.training.contrastive_ramp_epochs)",
    "--minimum-full-stage-epochs", "$($Config.experiment.training.minimum_full_stage_epochs)",
    "--patience", "$($Config.experiment.training.patience)",
    "--use-amp",
    "--n-fft", "$($Config.experiment.model.n_fft)",
    "--hop-length", "$($Config.experiment.model.hop_length)",
    "--spectral-channels", "$($Config.experiment.model.spectral_channels)",
    "--embedding-dim", "$($Config.experiment.model.embedding_dim)",
    "--environment-dim", "$($Config.experiment.model.environment_dim)",
    "--dropout", "$($Config.experiment.model.dropout)",
    "--bootstrap-draws", "$($Config.experiment.statistics.bootstrap_draws)",
    "--bootstrap-seed", "$($Config.experiment.statistics.bootstrap_seed)"
)

function Show-Command {
    param([string[]]$Arguments)
    $Quoted = $Arguments | ForEach-Object {
        if ($_ -match "\s") { '"{0}"' -f $_ } else { $_ }
    }
    Write-Host ("{0} {1}" -f $Python, ($Quoted -join " "))
}

if ($WriteRelease -and -not $Execute) {
    throw "WriteRelease is invalid without Execute."
}
if ($Execute -and $Stage -eq "all") {
    throw (
        "Direct -Stage all execution is disabled because macro generation " +
        "must occur between experiment and release. Use " +
        "run_all_local_after_gpu_free.ps1 -Plan formal."
    )
}
if ($WriteRelease -and $Stage -notin @("all", "release")) {
    throw "WriteRelease is valid only for the all or release stage."
}
if ($WriteRelease -and -not $MacroValues) {
    throw "WriteRelease requires an explicit audited MacroValues path."
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

Set-Location -LiteralPath $ProjectRoot
Write-Host "Project root: $ProjectRoot"
Write-Host "Frozen config: $ConfigPath"
Write-Host "Execution requested: $Execute"

if (-not $Execute) {
    if ($Stage -in @("all", "cache")) {
        if (Test-Path -LiteralPath $CacheRoot) {
            Write-Host (
                "Formal cache exists. Execute would fail closed unless " +
                "-AllowValidatedReuse and its pinned manifest SHA-256 are supplied: " +
                "$CacheRoot"
            )
        } else {
            Show-Command -Arguments $CacheArguments
        }
    }

    if ($Stage -in @("all", "experiment")) {
        if (-not (Test-Path -LiteralPath $CacheManifest -PathType Leaf)) {
            Write-Host "Experiment is gated: formal cache manifest is absent."
        } elseif (Test-Path -LiteralPath $RunRoot) {
            Write-Host (
                "Formal run exists. Execute would fail closed unless " +
                "-AllowValidatedReuse and pinned cache/run SHA-256 values are " +
                "supplied: $RunRoot"
            )
        } else {
            Show-Command -Arguments $ExperimentArguments
        }
    }

    if ($Stage -in @("all", "release")) {
        $ReleaseScript = Join-Path $PackageRoot "validate_release.py"
        if (-not (Test-Path -LiteralPath $ReleaseScript -PathType Leaf)) {
            Write-Host "Release validator is absent; release remains closed."
        } else {
            $PaperRoot = Join-Path $ProjectRoot "paper"
            $ReleaseLock = Join-Path $PaperRoot "release_lock.json"
            if (-not (Test-Path -LiteralPath $RunJson -PathType Leaf)) {
                Write-Host "Release is gated: formal run.json is absent at $RunJson"
            } elseif ($MacroValues) {
                $ResolvedMacroValues = [System.IO.Path]::GetFullPath(
                    (Join-Path $ProjectRoot $MacroValues)
                )
                $ReleaseArguments = @(
                    $ReleaseScript,
                    "--run-json", $RunJson,
                    "--paper-root", $PaperRoot,
                    "--macro-values", $ResolvedMacroValues
                )
                Show-Command -Arguments $ReleaseArguments
            } elseif (Test-Path -LiteralPath $ReleaseLock -PathType Leaf) {
                $ReleaseArguments = @(
                    $ReleaseScript,
                    "--run-json", $RunJson,
                    "--paper-root", $PaperRoot
                )
                Show-Command -Arguments $ReleaseArguments
            } else {
                Write-Host "Release remains closed: generate an audited macro manifest, then pass -MacroValues."
                Write-Host "Add -WriteRelease only after manifest preflight succeeds."
            }
        }
    }

    Write-Host "Dry preflight complete. No cache build, model training, or release write was started."
    exit 0
}

Assert-ExecutionAcknowledged
Enter-ExecutionMutex
try {
    $CacheDigest = ""
    $RunDigest = ""
    $CacheBuiltThisInvocation = $false
    $RunBuiltThisInvocation = $false

    if ($Stage -in @("all", "cache")) {
        Assert-MachineIdle
        if (Test-Path -LiteralPath $CacheRoot) {
            $CacheDigest = Assert-CacheReuse
        } else {
            Show-Command -Arguments $CacheArguments
            & $Python @CacheArguments
            if ($LASTEXITCODE -ne 0) {
                throw "Formal cache build failed."
            }
            if (-not (Test-Path -LiteralPath $CacheManifest -PathType Leaf)) {
                throw "Formal cache build did not produce manifest.json."
            }
            $CacheDigest = (
                Get-FileHash -Algorithm SHA256 -LiteralPath $CacheManifest
            ).Hash.ToLowerInvariant()
            $CacheBuiltThisInvocation = $true
        }
    }

    if ($Stage -in @("all", "experiment")) {
        if (-not (Test-Path -LiteralPath $CacheManifest -PathType Leaf)) {
            throw "Experiment is gated: formal cache manifest is absent."
        }
        if (-not $CacheBuiltThisInvocation) {
            $CacheDigest = Assert-CacheReuse
        }
        if (Test-Path -LiteralPath $RunRoot) {
            $RunDigest = Assert-RunReuse
        } else {
            Assert-MachineIdle
            Show-Command -Arguments $ExperimentArguments
            & $Python @ExperimentArguments
            if ($LASTEXITCODE -ne 0) {
                throw "Formal experiment failed."
            }
            if (-not (Test-Path -LiteralPath $RunJson -PathType Leaf)) {
                throw "Formal experiment did not produce run.json."
            }
            Assert-RunRecordEligible
            $RunDigest = (
                Get-FileHash -Algorithm SHA256 -LiteralPath $RunJson
            ).Hash.ToLowerInvariant()
            $RunBuiltThisInvocation = $true
        }
    }

    if ($Stage -in @("all", "release")) {
        $ReleaseScript = Join-Path $PackageRoot "validate_release.py"
        if (-not (Test-Path -LiteralPath $ReleaseScript -PathType Leaf)) {
            throw "Release validator is absent; release remains closed."
        }
        if (-not (Test-Path -LiteralPath $CacheManifest -PathType Leaf)) {
            throw "Release is gated: formal cache manifest is absent."
        }
        if (-not $CacheBuiltThisInvocation -and -not $CacheDigest) {
            $CacheDigest = Assert-CacheReuse
        }
        if (-not (Test-Path -LiteralPath $RunJson -PathType Leaf)) {
            throw "Release is gated: formal run.json is absent at $RunJson"
        }
        if (-not $RunBuiltThisInvocation -and -not $RunDigest) {
            $RunDigest = Assert-RunReuse
        }

        $PaperRoot = Join-Path $ProjectRoot "paper"
        $ReleaseLock = Join-Path $PaperRoot "release_lock.json"
        if ($MacroValues) {
            $ResolvedMacroValues = [System.IO.Path]::GetFullPath(
                (Join-Path $ProjectRoot $MacroValues)
            )
            $ReleaseArguments = @(
                $ReleaseScript,
                "--run-json", $RunJson,
                "--paper-root", $PaperRoot,
                "--macro-values", $ResolvedMacroValues
            )
            if ($WriteRelease) {
                $ReleaseArguments += "--write"
            }
        } elseif (Test-Path -LiteralPath $ReleaseLock -PathType Leaf) {
            $ReleaseArguments = @(
                $ReleaseScript,
                "--run-json", $RunJson,
                "--paper-root", $PaperRoot
            )
        } else {
            throw (
                "Release remains closed: generate an audited macro manifest " +
                "and pass -MacroValues."
            )
        }
        Assert-MachineIdle
        Show-Command -Arguments $ReleaseArguments
        & $Python @ReleaseArguments
        if ($LASTEXITCODE -ne 0) {
            throw "Release validation failed."
        }
    }
} finally {
    Exit-ExecutionMutex
}

Write-Host "Requested formal stage completed without controlling any pre-existing process."
