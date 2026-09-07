param(
    [switch]$Execute,
    [switch]$AllowValidatedReuse,
    [string]$Expected10kManifestSha256 = "",
    [string]$Expected30kManifestSha256 = "",
    [string]$Expected100kManifestSha256 = "",
    [string]$Acknowledgement = "",
    [string]$Python = "D:\Python\python.exe",
    [int]$MatlabBatchSize = 8192,
    [int]$MinimumFreeGpuMiB = 6000,
    [int]$MinimumFreeDiskGiB = 40
)

$forward = @{
    Execute = $Execute
    AllowValidatedReuse = $AllowValidatedReuse
    Expected10kManifestSha256 = $Expected10kManifestSha256
    Expected30kManifestSha256 = $Expected30kManifestSha256
    Expected100kManifestSha256 = $Expected100kManifestSha256
    Acknowledgement = $Acknowledgement
    Python = $Python
    Freeze = (Join-Path $PSScriptRoot "configs\formal_tvt_freeze_v3_8gb.json")
    MatlabBatchSize = $MatlabBatchSize
    MinimumFreeGpuMiB = $MinimumFreeGpuMiB
    MinimumFreeDiskGiB = $MinimumFreeDiskGiB
}

& (Join-Path $PSScriptRoot "run_v2_after_gpu_free.ps1") @forward
exit $LASTEXITCODE
