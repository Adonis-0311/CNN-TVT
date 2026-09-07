param(
    [switch]$Execute,
    [switch]$AllowValidatedReuse,
    [string]$Expected10kManifestSha256 = "",
    [string]$Expected30kManifestSha256 = "",
    [string]$Expected100kManifestSha256 = "",
    [string]$Acknowledgement = "",
    [string]$Python = "D:\Python\python.exe",
    [string]$Freeze = (Join-Path $PSScriptRoot "configs\formal_tvt_freeze_v3_8gb.json"),
    [int]$MatlabBatchSize = 8192,
    [int]$MinimumFreeGpuMiB = 6000,
    [int]$MinimumFreeDiskGiB = 40
)

$ErrorActionPreference = "Stop"
$RequiredAcknowledgement = "START_TVT_ONLY_WHEN_MACHINE_IS_IDLE"
$MutexName = "Global\VIMD_AMC_TVT_LOCAL_EXECUTION_V3_8GB"
$RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Cache10k = Join-Path $RepositoryRoot "standards\cache_factor_learning_10k_1024_v2"
$Cache30k = Join-Path $RepositoryRoot "standards\cache_factor_learning_30k_1024_v2"
$Cache100k = Join-Path $RepositoryRoot "standards\cache_factor_headline_1024_v2"
$Artifacts = Join-Path $RepositoryRoot "artifacts"
$LearningEvidence = Join-Path $Artifacts "tvt_learning_curve_v3_8gb\learning_curve_evidence.json"
$RunJson = Join-Path $Artifacts "tvt_headline_1024_10seed_v3_8gb\run.json"
$MacroValues = Join-Path $RepositoryRoot "tvt_submission\formal_macro_values_v3_8gb.json"
$ExecutionMutex = $null

function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments, [string]$Failure)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw $Failure
    }
}

function Invoke-JsonChecked {
    param(
        [string]$Program,
        [string[]]$Arguments,
        [string]$Failure,
        [switch]$Quiet
    )
    $Output = @(& $Program @Arguments)
    if ($LASTEXITCODE -ne 0) {
        throw $Failure
    }
    $Text = ($Output -join "`n").Trim()
    try {
        $Payload = $Text | ConvertFrom-Json
    } catch {
        throw "$Failure (command did not return valid JSON)"
    }
    if (-not $Quiet) {
        Write-Host $Text
    }
    return $Payload
}

function Assert-ExecutionSourceDigest {
    param([string]$ExpectedDigest, [string]$Stage, [string]$FreezePath)
    $Payload = Invoke-JsonChecked `
        -Program $Python `
        -Arguments @(
            "experiments\run_formal_tvt_v2.py", "--freeze", $FreezePath, "--preflight-only"
        ) `
        -Failure "TVT execution source integrity check failed at $Stage" `
        -Quiet
    $ActualDigest = $Payload.source_tree_binding.aggregate_digest
    Assert-Sha256 -Value $ActualDigest -Label "$Stage source binding"
    if ($ActualDigest -ne $ExpectedDigest) {
        throw "TVT execution sources changed at $Stage."
    }
    Write-Host (
        [ordered]@{
            schema_version = "vimd_amc.tvt.v3_8gb_source_checkpoint.v1"
            ok = $true
            stage = $Stage
            source_tree_aggregate_digest = $ActualDigest
        } | ConvertTo-Json -Compress
    )
}

function Assert-Sha256 {
    param([string]$Value, [string]$Label)
    if ($Value -notmatch "^[0-9a-fA-F]{64}$") {
        throw "$Label must be an exact 64-hex SHA-256 digest."
    }
}

function Get-Sha256FileDigest {
    param([string]$LiteralPath)
    # Older Windows PowerShell installations can lack Get-FileHash. Use the
    # .NET SHA-256 implementation so this cache-reuse evidence gate remains
    # available on every supported local PowerShell runtime.
    $Stream = [System.IO.File]::OpenRead($LiteralPath)
    try {
        $Hasher = [System.Security.Cryptography.SHA256]::Create()
        try {
            return (($Hasher.ComputeHash($Stream) | ForEach-Object { $_.ToString("x2") }) -join "")
        }
        finally {
            $Hasher.Dispose()
        }
    }
    finally {
        $Stream.Dispose()
    }
}

function Assert-ReusableCache {
    param([string]$CacheRoot, [string]$ExpectedSha256)
    if (-not $AllowValidatedReuse) {
        throw "Existing cache reuse is disabled: $CacheRoot"
    }
    Assert-Sha256 -Value $ExpectedSha256 -Label "$CacheRoot manifest digest"
    $Manifest = Join-Path $CacheRoot "manifest.json"
    if (-not (Test-Path -LiteralPath $Manifest -PathType Leaf)) {
        throw "Existing cache directory is incomplete: $CacheRoot"
    }
    $Actual = (Get-Sha256FileDigest -LiteralPath $Manifest).ToLower()
    if ($Actual -ne $ExpectedSha256.ToLower()) {
        throw "Existing cache manifest digest mismatch: $CacheRoot"
    }
}

function Assert-MachineIdle {
    $Active = @(
        Get-CimInstance Win32_Process |
            Where-Object {
                $_.Name -match "^(python|pythonw|matlab|matlabworker)(\.exe|\.bin)?$"
            } |
            Select-Object ProcessId, Name, CommandLine
    )
    if ($Active.Count -gt 0) {
        $Summary = $Active | ForEach-Object {
            "PID=$($_.ProcessId) Name=$($_.Name) Command=$($_.CommandLine)"
        }
        throw "Python/MATLAB work is active. Nothing was started.`n$($Summary -join "`n")"
    }
    $NvidiaSmi = Get-Command "nvidia-smi" -ErrorAction SilentlyContinue
    if (-not $NvidiaSmi) {
        throw "nvidia-smi is unavailable; GPU idle state cannot be proven."
    }
    $FreeValues = @(
        & $NvidiaSmi.Source --query-gpu=memory.free --format=csv,noheader,nounits |
            ForEach-Object {
                $Value = 0
                if (-not [int]::TryParse($_.Trim(), [ref]$Value)) {
                    throw "Could not parse nvidia-smi output: $_"
                }
                $Value
            }
    )
    if ($LASTEXITCODE -ne 0 -or $FreeValues.Count -eq 0) {
        throw "GPU memory query failed."
    }
    $MinimumObserved = ($FreeValues | Measure-Object -Minimum).Minimum
    if ($MinimumObserved -lt $MinimumFreeGpuMiB) {
        throw "Only $MinimumObserved MiB GPU memory is free; $MinimumFreeGpuMiB MiB is required."
    }
    $DriveRoot = [System.IO.Path]::GetPathRoot($RepositoryRoot)
    $FreeBytes = [System.IO.DriveInfo]::new($DriveRoot).AvailableFreeSpace
    if ($FreeBytes -lt ([int64]$MinimumFreeDiskGiB * 1GB)) {
        throw "Less than $MinimumFreeDiskGiB GiB is free on $DriveRoot."
    }
}

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable not found: $Python"
}
if (-not (Test-Path -LiteralPath $Freeze -PathType Leaf)) {
    throw "Formal freeze is missing: $Freeze"
}
Push-Location $RepositoryRoot
try {
    $Plans = @(
        @{ Path = $Cache10k; Preset = "learning_10k_v2"; Hash = $Expected10kManifestSha256 },
        @{ Path = $Cache30k; Preset = "learning_30k_v2"; Hash = $Expected30kManifestSha256 },
        @{ Path = $Cache100k; Preset = "headline_v2"; Hash = $Expected100kManifestSha256 }
    )
    $FreezePreflight = Invoke-JsonChecked $Python @(
        "tvt_submission\validate_formal_freeze_v2.py", "--freeze", $Freeze
    ) "formal freeze validation failed"
    $LearningPreflight = Invoke-JsonChecked $Python @(
        "experiments\run_learning_curve_v2.py", "--freeze", $Freeze, "--preflight-only"
    ) "learning-curve preflight failed"
    $FormalPreflight = Invoke-JsonChecked $Python @(
        "experiments\run_formal_tvt_v2.py", "--freeze", $Freeze, "--preflight-only"
    ) "formal v2 preflight failed"
    if (
        $LearningPreflight.device_contract.matches -ne $true -or
        $FormalPreflight.device_contract.matches -ne $true -or
        $LearningPreflight.device_contract.frozen -ne "cuda" -or
        $FormalPreflight.device_contract.frozen -ne "cuda"
    ) {
        throw "TVT frozen CUDA device contract did not pass."
    }
    $CudaExecution = $FormalPreflight.cuda_execution
    if (
        $null -eq $CudaExecution -or
        [int]$CudaExecution.minimum_free_gpu_mib -ne $MinimumFreeGpuMiB -or
        [int]$CudaExecution.per_device_batch_size -ne 32 -or
        $CudaExecution.use_amp -ne $true -or
        [string]::IsNullOrWhiteSpace([string]$CudaExecution.allocator_configuration)
    ) {
        throw "CUDA memory settings differ from the frozen 8 GiB execution amendment."
    }
    $LearningSourceDigest = $LearningPreflight.source_tree_binding.aggregate_digest
    $FormalSourceDigest = $FormalPreflight.source_tree_binding.aggregate_digest
    Assert-Sha256 -Value $LearningSourceDigest -Label "learning source binding"
    Assert-Sha256 -Value $FormalSourceDigest -Label "formal source binding"
    if ($LearningSourceDigest -ne $FormalSourceDigest) {
        throw "Learning and formal source-tree bindings disagree."
    }
    $IntegritySummary = [ordered]@{
        schema_version = "vimd_amc.tvt.v3_8gb_static_execution_integrity.v1"
        ok = $true
        execution_started = $false
        frozen_device = "cuda"
        source_tree_profile = $FormalPreflight.source_tree_binding.profile
        source_tree_aggregate_digest = $FormalSourceDigest
        source_tree_file_count = $FormalPreflight.source_tree_binding.file_count
        freeze_sha256 = $FreezePreflight.freeze_sha256
        minimum_free_gpu_mib = [int]$CudaExecution.minimum_free_gpu_mib
        per_device_batch_size = [int]$CudaExecution.per_device_batch_size
    }
    Write-Host ($IntegritySummary | ConvertTo-Json -Compress)
    foreach ($Plan in $Plans) {
        Invoke-Checked $Python @(
            "standards\build_factor_cache.py",
            "--output", $Plan.Path,
            "--preset", $Plan.Preset,
            "--sample-length", "1024",
            "--guard-samples", "96",
            "--master-seed", "20260727",
            "--matlab-batch-size", "$MatlabBatchSize",
            "--print-policy-only"
        ) "cache execution preflight failed: $($Plan.Preset)"
    }
    if (-not $Execute) {
        Write-Host (
            "V3 8 GiB dry preflight complete, including all three cache-size " +
            "policies. No cache, MATLAB, training, artifact, or release " +
            "write was started."
        )
        exit 0
    }
    if ($Acknowledgement -ne $RequiredAcknowledgement) {
        throw "Pass -Acknowledgement '$RequiredAcknowledgement' only after confirming the machine is idle."
    }
    $CreatedNew = $false
    $ExecutionMutex = [System.Threading.Mutex]::new(
        $false,
        $MutexName,
        [ref]$CreatedNew
    )
    if (-not $ExecutionMutex.WaitOne(0)) {
        throw "Another TVT v2 execution owns mutex $MutexName."
    }
    Assert-MachineIdle
    $env:PYTORCH_CUDA_ALLOC_CONF = [string]$CudaExecution.allocator_configuration

    foreach ($Plan in $Plans) {
        if (Test-Path -LiteralPath $Plan.Path) {
            Assert-ReusableCache -CacheRoot $Plan.Path -ExpectedSha256 $Plan.Hash
        } else {
            Invoke-Checked $Python @(
                "standards\build_factor_cache.py",
                "--output", $Plan.Path,
                "--preset", $Plan.Preset,
                "--sample-length", "1024",
                "--guard-samples", "96",
                "--master-seed", "20260727",
                "--matlab-timeout-s", "3600",
                "--matlab-batch-size", "$MatlabBatchSize"
            ) "cache build failed: $($Plan.Path)"
        }
    }
    Assert-ExecutionSourceDigest `
        -ExpectedDigest $FormalSourceDigest `
        -Stage "after_cache_builds" `
        -FreezePath $Freeze
    Invoke-Checked $Python @(
        "experiments\run_learning_curve_v2.py",
        "--freeze", $Freeze,
        "--cache-10k", $Cache10k,
        "--cache-30k", $Cache30k,
        "--cache-100k", $Cache100k,
        "--validate-caches-only"
    ) "cache contract or shared-validation-source validation failed"
    Assert-MachineIdle
    Invoke-Checked $Python @(
        "experiments\run_learning_curve_v2.py",
        "--freeze", $Freeze,
        "--cache-10k", $Cache10k,
        "--cache-30k", $Cache30k,
        "--cache-100k", $Cache100k,
        "--device", "cuda",
        "--output", $Artifacts,
        "--evidence-output", $LearningEvidence
    ) "learning-curve execution or evidence binding failed"
    Assert-ExecutionSourceDigest `
        -ExpectedDigest $FormalSourceDigest `
        -Stage "after_learning_curve" `
        -FreezePath $Freeze
    Assert-MachineIdle
    Invoke-Checked $Python @(
        "experiments\run_formal_tvt_v2.py",
        "--freeze", $Freeze,
        "--cache-root", $Cache100k,
        "--learning-curve-evidence", $LearningEvidence,
        "--device", "cuda",
        "--output", $Artifacts
    ) "formal v2 execution or scientific release gate failed"
    Invoke-Checked $Python @(
        "tvt_submission\validate_v2_paper_release.py",
        "--freeze", $Freeze,
        "--run-json", $RunJson,
        "--learning-curve-evidence", $LearningEvidence,
        "--macro-values", $MacroValues,
        "--write-macro-manifest"
    ) "formal v2 paper macro-manifest generation failed"
    Write-Host (
        "Formal machine evidence and the artifact-bound V3 8 GiB macro manifest " +
        "were generated. Human review is still required before the explicit " +
        "paper release write."
    )
}
finally {
    if ($ExecutionMutex) {
        try {
            $ExecutionMutex.ReleaseMutex()
        } catch {
        }
        $ExecutionMutex.Dispose()
    }
    Pop-Location
}
