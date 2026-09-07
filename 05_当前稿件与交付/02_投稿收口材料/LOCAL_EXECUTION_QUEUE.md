# TVT v3 8 GiB local execution queue

Status date: 2026-08-02
Authority: `configs/formal_tvt_freeze_v3_8gb.json`, which hash-binds the v2
data/model/statistical base and changes only the pre-result CUDA resource plan.

This is the operator-facing entry point for all remaining machine computation.
The only supported fresh execution command is
`run_v3_8gb_after_gpu_free.ps1`. The underlying `run_v2_after_gpu_free.ps1`
is a compatibility implementation, not the operator entry point. Older `run_all_local_after_gpu_free.ps1`,
`run_local.ps1`, v1/v2 freeze execution, five-seed, and 55-fit instructions
are historical provenance and are not valid TVT v3 8 GiB evidence.

## 1. Safe dry preflight

From the repository root:

```powershell
Set-Location -LiteralPath "D:\CNN信号调制识别\TVT支线\02_实验源码与证据\01_当前可复现主线"
& .\tvt_submission\run_v3_8gb_after_gpu_free.ps1 `
  -Python "D:\Python\python.exe"
```

This validates the frozen v3 8 GiB contract and both runner preflights. It starts no
cache generation, MATLAB simulation, training, scientific artifact, macro
manifest, or paper release write.

## 2. The single fresh execution command

Only after the human operator has confirmed that Python/MATLAB work is idle,
the GPU has at least 6,000 MiB free, and the project volume has at least
40 GiB free. The frozen plan uses AMP, a per-device batch size of 32, one CUDA
fit at a time, and the pinned PyTorch allocator setting:

```powershell
Set-Location -LiteralPath "D:\CNN信号调制识别\TVT支线\02_实验源码与证据\01_当前可复现主线"
& .\tvt_submission\run_v3_8gb_after_gpu_free.ps1 `
  -Execute `
  -Acknowledgement "START_TVT_ONLY_WHEN_MACHINE_IS_IDLE" `
  -MinimumFreeGpuMiB 6000 `
  -MinimumFreeDiskGiB 40 `
  -Python "D:\Python\python.exe"
```

The command is sequential. It does not launch 120 fits concurrently and it
does not terminate, pause, or take ownership of unrelated work. Keep the
machine idle until the command exits. Do not change source, freeze, models,
seeds, splits, optimizer, checkpoint, bootstrap, or gate settings after
opening an outcome.

## 3. What the wrapper performs

In order, the wrapper:

1. validates the hash-bound v3 8 GiB amendment and lower-level preflights;
2. acquires `Global\VIMD_AMC_TVT_LOCAL_EXECUTION_V3_8GB`;
3. enforces Python/MATLAB, GPU-memory, and disk-space gates;
4. builds the 10k, 30k, and 100k MATLAB TDL caches;
5. checks cache designation, integrity, source disjointness, and identical
   frozen validation identities across scales;
6. trains A0/A5 at all three scales using five fixed seeds and writes the
   learning-curve evidence;
7. runs the 12-model × 10-seed formal grid at the fixed 100k scale;
8. derives and writes the fail-closed scientific gate; and
9. derives and writes the source-bound v3 8 GiB paper-macro manifest.

Steps 8–9 can succeed only when the run is complete and all required
runner-native artifacts pass. The wrapper never writes release macros into the
manuscript and never creates a release lock.

## 4. Expected outputs

| Stage | Required output |
|---|---|
| 10k cache | `standards/cache_factor_learning_10k_1024_v2/manifest.json` |
| 30k cache | `standards/cache_factor_learning_30k_1024_v2/manifest.json` |
| 100k cache | `standards/cache_factor_headline_1024_v2/manifest.json` |
| Learning curve | `artifacts/tvt_learning_curve_v3_8gb/learning_curve_evidence.json` |
| Formal run | `artifacts/tvt_headline_1024_10seed_v3_8gb/run.json` |
| Scientific gate | `artifacts/tvt_headline_1024_10seed_v3_8gb/v2_scientific_release_gate.json` |
| Paper macro manifest | `tvt_submission/formal_macro_values_v3_8gb.json` |

The formal run also contains per-fit result/checkpoint/history files,
source-aligned prediction NPZs for every required regime, statistical CSV/JSON
artifacts, complexity/latency evidence, and the occupancy diagnostic. Do not
judge completion by directory presence. The wrapper must exit zero and the
validators must agree exactly on hashes and content.

## 5. Validated continuation after interruption

The default behavior rejects every existing cache directory. Reuse is allowed
only after the operator audits each existing cache and pins its exact manifest
SHA-256:

```powershell
$Sha10k = (Get-FileHash -Algorithm SHA256 `
  .\standards\cache_factor_learning_10k_1024_v2\manifest.json).Hash.ToLowerInvariant()
$Sha30k = (Get-FileHash -Algorithm SHA256 `
  .\standards\cache_factor_learning_30k_1024_v2\manifest.json).Hash.ToLowerInvariant()
$Sha100k = (Get-FileHash -Algorithm SHA256 `
  .\standards\cache_factor_headline_1024_v2\manifest.json).Hash.ToLowerInvariant()

& .\tvt_submission\run_v3_8gb_after_gpu_free.ps1 `
  -Execute `
  -AllowValidatedReuse `
  -Expected10kManifestSha256 $Sha10k `
  -Expected30kManifestSha256 $Sha30k `
  -Expected100kManifestSha256 $Sha100k `
  -Acknowledgement "START_TVT_ONLY_WHEN_MACHINE_IS_IDLE" `
  -MinimumFreeGpuMiB 6000 `
  -MinimumFreeDiskGiB 40 `
  -Python "D:\Python\python.exe"
```

This switch covers only exact validated cache reuse. Existing or partial
learning/formal output directories are fail-closed and must be audited rather
than overwritten. Never delete a partial run merely to hide an interruption;
preserve it for diagnosis and select a new clean output only through a
prospective contract change.

## 6. Post-run read-only verification

After a zero exit, re-run the paper bridge without writing:

```powershell
& "D:\Python\python.exe" .\tvt_submission\validate_v2_paper_release.py `
  --freeze .\tvt_submission\configs\formal_tvt_freeze_v3_8gb.json `
  --run-json .\artifacts\tvt_headline_1024_10seed_v3_8gb\run.json `
  --learning-curve-evidence `
    .\artifacts\tvt_learning_curve_v3_8gb\learning_curve_evidence.json `
  --macro-values .\tvt_submission\formal_macro_values_v3_8gb.json `
  --validate-macro-manifest
```

The JSON result must report `"ok": true`,
`"action": "v2_macro_manifest_validated"`, `"write_performed": false`,
`"submission_unlocked": false`, and the expected run/macro binding. This action
rederives and reads the existing manifest; the no-action preflight does not
validate an existing `--macro-values` file. A passing scientific gate still
does not authorize a positive claim that its individual branch or mechanism
result does not support.

## 7. Human-approved release write

First inspect:

- the learning curve and all 120 fit statuses;
- the three simultaneous confirmatory intervals;
- every OOD, receiver-stress, and clean-retention branch;
- the occupancy/gain result and its allowed claim language;
- exploratory contrasts, baseline labels, complexity, and latency;
- the exact macro manifest sources and hashes.

After a human author approves the evidence and claim scope, perform the
explicit atomic release write:

```powershell
& "D:\Python\python.exe" .\tvt_submission\validate_v2_paper_release.py `
  --freeze .\tvt_submission\configs\formal_tvt_freeze_v3_8gb.json `
  --run-json .\artifacts\tvt_headline_1024_10seed_v3_8gb\run.json `
  --learning-curve-evidence `
    .\artifacts\tvt_learning_curve_v3_8gb\learning_curve_evidence.json `
  --macro-values .\tvt_submission\formal_macro_values_v3_8gb.json `
  --paper-root .\paper `
  --write-release
```

This is intentionally separate from training. It writes
`paper/results_auto.tex` and `paper/release_lock.json` only if every binding is
intact. It must not be run with a placeholder, edited macro manifest, failing
scientific gate, or unreviewed claim outcome.

Immediately revalidate the written release against all still-present source
artifacts:

```powershell
& "D:\Python\python.exe" .\tvt_submission\validate_v2_paper_release.py `
  --freeze .\tvt_submission\configs\formal_tvt_freeze_v3_8gb.json `
  --run-json .\artifacts\tvt_headline_1024_10seed_v3_8gb\run.json `
  --learning-curve-evidence `
    .\artifacts\tvt_learning_curve_v3_8gb\learning_curve_evidence.json `
  --macro-values .\tvt_submission\formal_macro_values_v3_8gb.json `
  --paper-root .\paper `
  --validate-release
```

The JSON result must report `"ok": true`,
`"action": "v2_existing_release_validated"`, and both unlock fields `true`.

Select the public LaTeX branch only after authorship metadata is complete, then
compile and validate the release PDF:

```powershell
& "D:\Python\python.exe" .\tvt_submission\validate_paper_build.py `
  --mode release `
  --paper-root .\paper `
  --log .\paper\build_release\main.log `
  --pdf .\paper\build_release\main.pdf `
  --fls .\paper\build_release\main.fls
```

The exact LaTeX build command/environment remains operator-controlled; the
validator fails on internal mode, placeholders, authorship placeholders,
undefined citations/references, fatal errors, overfull boxes, stale
dependencies, or a release-lock mismatch.

## 8. Human and external boundaries

No automation may supply or approve author names, affiliations, funding,
conflicts, patent timing, license/citation verification, current TVT policy,
or the final submission decision. The current initial draft already makes no
native RadioML-transfer, MFENet-reproduction, or superiority claim, so those
external materials are not inputs to this queue. Adding such a claim later
requires authorized material/contact and a separate prospective protocol.

Do not promote screening, diagnostic, oracle-control, historical v1, or
manually entered numbers. Do not call a passing scientific gate “submission
unlocked.” Do not write a public PDF until the v2 release lock and strict
release build both pass.
