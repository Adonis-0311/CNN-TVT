param(
    [switch]$Execute,
    [ValidateSet("all", "cache", "experiment", "release")]
    [string]$Stage = "all",
    [string]$MacroValues = "",
    [switch]$WriteRelease,
    [string]$Python = "D:\Python\python.exe"
)

$ErrorActionPreference = "Stop"
$PackageRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $PackageRoot
$ConfigPath = Join-Path $PackageRoot "configs\formal_tvt_freeze_v1.json"
$ValidatorPath = Join-Path $PackageRoot "validate_formal_freeze.py"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable not found: $Python"
}
if (-not (Test-Path -LiteralPath $ValidatorPath -PathType Leaf)) {
    throw "Formal-freeze validator not found: $ValidatorPath"
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

Set-Location -LiteralPath $ProjectRoot
Write-Host "Project root: $ProjectRoot"
Write-Host "Frozen config: $ConfigPath"
Write-Host "Execution requested: $Execute"

if ($Stage -in @("all", "cache")) {
    if (Test-Path -LiteralPath $CacheRoot) {
        Write-Host "Formal cache already exists; refusing to overwrite: $CacheRoot"
    } else {
        Show-Command -Arguments $CacheArguments
        if ($Execute) {
            & $Python @CacheArguments
            if ($LASTEXITCODE -ne 0) { throw "Formal cache build failed." }
        }
    }
}

if ($Stage -in @("all", "experiment")) {
    if (-not (Test-Path -LiteralPath (Join-Path $CacheRoot "manifest.json"))) {
        Write-Host "Experiment is gated: formal cache manifest is absent."
    } elseif (Test-Path -LiteralPath $RunRoot) {
        Write-Host "Run directory already exists; refusing to merge or overwrite: $RunRoot"
    } else {
        Show-Command -Arguments $ExperimentArguments
        if ($Execute) {
            & $Python @ExperimentArguments
            if ($LASTEXITCODE -ne 0) { throw "Formal experiment failed." }
        }
    }
}

if ($Stage -in @("all", "release")) {
    $ReleaseScript = Join-Path $PackageRoot "validate_release.py"
    if (-not (Test-Path -LiteralPath $ReleaseScript -PathType Leaf)) {
        Write-Host "Release validator is not yet present; release remains closed."
    } else {
        $RunJson = Join-Path $RunRoot "run.json"
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
            if ($WriteRelease) {
                $ReleaseArguments += "--write"
            }
            Show-Command -Arguments $ReleaseArguments
            if ($Execute) {
                & $Python @ReleaseArguments
                if ($LASTEXITCODE -ne 0) { throw "Release validation failed." }
            }
        } elseif (Test-Path -LiteralPath $ReleaseLock -PathType Leaf) {
            $ReleaseArguments = @(
                $ReleaseScript,
                "--run-json", $RunJson,
                "--paper-root", $PaperRoot
            )
            Show-Command -Arguments $ReleaseArguments
            if ($Execute) {
                & $Python @ReleaseArguments
                if ($LASTEXITCODE -ne 0) { throw "Existing release validation failed." }
            }
        } else {
            Write-Host "Release remains closed: generate an audited macro manifest, then pass -MacroValues."
            Write-Host "Add -WriteRelease only after manifest preflight succeeds."
        }
    }
}

if (-not $Execute) {
    Write-Host "Dry preflight complete. No cache build, model training, or release write was started."
}
