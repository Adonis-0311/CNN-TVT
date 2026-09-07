param(
    [switch]$Execute,
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
$Script = "diagnostics\run_iqformer_route_v4_diagnostic.py"
$Preregistration = "docs\VIMD_V4_DSBN_PREREGISTRATION.md"
$Output = "artifacts\diagnostic_iqformer_route_v4_dsbn"
$ExecutionMutex = $null
$Arguments = @(
    $Script,
    "--cache-root", "standards\cache_factor_screening_1024_v1",
    "--output", $Output,
    "--cpu-threads", "1",
    "--execute-preregistered-diagnostic"
)

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable not found: $Python"
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

Set-Location -LiteralPath $ProjectRoot
if (-not (Test-Path -LiteralPath $Script -PathType Leaf)) {
    throw "Candidate diagnostic script is absent: $Script"
}
if (-not (Test-Path -LiteralPath $Preregistration -PathType Leaf)) {
    throw "Candidate preregistration is absent: $Preregistration"
}

if (-not $Execute) {
    $Display = "$Python " + ($Arguments -join " ")
    Write-Host $Display
    Write-Host "Dry preflight complete. VIMD-v4 training was not started."
    Write-Host "The diagnostic is one-shot and non-evidence. Read docs\VIMD_V4_DSBN_PREREGISTRATION.md before using -Execute."
    exit 0
}

Assert-ExecutionAcknowledged
Enter-ExecutionMutex
try {
    Assert-MachineIdle
    if (Test-Path -LiteralPath $Output) {
        throw "Candidate output already exists; refusing to merge, overwrite, or tune after outcome."
    }
    $Display = "$Python " + ($Arguments -join " ")
    Write-Host $Display
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "The preregistered VIMD-v4 diagnostic failed operationally."
    }
} finally {
    Exit-ExecutionMutex
}

Write-Host "Diagnostic finished. Its output remains non-evidence and must be judged only by the frozen all-required gate."
