# TVT Local Simulation and Training Queue

Status date: 2026-07-28

This is the operator-facing entry point for every planned local long-running
job. Nothing in this handoff was started by the delivery turn. The canonical
wrapper is `run_all_local_after_gpu_free.ps1`; it is dry-run by default and
never kills, pauses, resumes, waits for, or signals another process.

## Plans

| Plan | Content | Paper evidence |
|---|---|---|
| `preflight` | Validate the formal freeze and print the formal and V4 commands | No |
| `formal` | Build the formal MATLAB TDL cache; run 11 models × 5 seeds; generate `tvt_submission/formal_macro_values.json`; perform release preflight | Only if every automatic gate passes |
| `release` | Revalidate an existing pinned formal cache/run and the audited macro manifest; optionally write the release with `-WriteRelease` | No new experiment |
| `candidate-v4` | Run the preregistered one-shot VIMD-v4 DSBN diagnostic on the screening cache | No |
| `all` | Run `formal`, then the separate V4 diagnostic | Formal part only; V4 stays non-evidence |

The formal plan is sequential. It does not launch 55 concurrent GPU jobs.
The V4 diagnostic is outside the frozen 11-model formal family.

## Mandatory execution gates

Every `-Execute` path in the wrapper and both lower-level runners requires:

- the exact acknowledgment
  `START_TVT_ONLY_WHEN_MACHINE_IS_IDLE`;
- the global non-preemptive mutex
  `Global\VIMD_AMC_TVT_LOCAL_EXECUTION_V1`;
- no active `python`, `pythonw`, `matlab`, `matlabworker`, `soffice.exe`, or
  `soffice.bin` process, including a process whose command line points inside
  this project;
- at least 7,000 MiB free GPU memory by default;
- at least 20 GiB free on the project volume by default.

The process/GPU/disk gate is repeated between formal stages. A failed or
abandoned mutex is fail-closed; no process is stopped and no lock is stolen.
The checks reduce collision risk but do not reserve the GPU against an
unrelated job launched later, so the operator must keep the machine idle for
the entire run.

Existing formal cache/run directories are rejected by default. Reuse requires
the explicit `-AllowValidatedReuse` switch and an operator-pinned SHA-256 for
each existing `manifest.json` or `run.json`. `run.json` reuse additionally
requires a complete, eligible 11×5 run matching the frozen reference model.

## Safe dry-run

```powershell
Set-Location -LiteralPath "D:\CNN信号调制识别\vimd_amc"

& .\tvt_submission\run_all_local_after_gpu_free.ps1 `
  -Plan preflight `
  -Python "D:\Python\python.exe"
```

Expected final line:

```text
Queue dry-run complete. No cache, simulation, training, candidate, macro, or release write was started.
```

## Fresh formal run after the machine becomes idle

```powershell
& .\tvt_submission\run_all_local_after_gpu_free.ps1 `
  -Plan formal `
  -Execute `
  -Acknowledgement "START_TVT_ONLY_WHEN_MACHINE_IS_IDLE" `
  -MinimumFreeGpuMiB 7000 `
  -MinimumFreeDiskGiB 20 `
  -Python "D:\Python\python.exe"
```

This builds the formal cache, performs all 55 sequential fits, creates
`tvt_submission/formal_macro_values.json`, and runs release preflight. It
does not write `paper/results_auto.tex` or `paper/release_lock.json`.

## Release revalidation and human-approved write

After the formal command exits successfully, pin the two immutable inputs:

```powershell
$CacheSha = (
  Get-FileHash -Algorithm SHA256 `
    .\standards\cache_factor_headline_1024_v1\manifest.json
).Hash.ToLowerInvariant()
$RunSha = (
  Get-FileHash -Algorithm SHA256 `
    .\artifacts\tvt_headline_1024_5seed_v1\run.json
).Hash.ToLowerInvariant()
```

Revalidate without writing:

```powershell
& .\tvt_submission\run_all_local_after_gpu_free.ps1 `
  -Plan release `
  -Execute `
  -AllowValidatedReuse `
  -ExpectedCacheManifestSha256 $CacheSha `
  -ExpectedRunJsonSha256 $RunSha `
  -Acknowledgement "START_TVT_ONLY_WHEN_MACHINE_IS_IDLE" `
  -MinimumFreeGpuMiB 7000 `
  -MinimumFreeDiskGiB 20 `
  -Python "D:\Python\python.exe"
```

Only after the human release review, repeat that `release` command with
`-WriteRelease`. This writes the locked paper results; it does not rerun the
cache or the 55 fits.

## Explicit validated continuation

If a previously completed cache is to be reused for a still-absent run, pin
its manifest and add:

```powershell
-AllowValidatedReuse `
-ExpectedCacheManifestSha256 $CacheSha
```

to the `formal` command. If the run directory also exists, pin and supply
`-ExpectedRunJsonSha256 $RunSha`; otherwise execution fails closed. Never
reuse a partial, invalidated, source-mutated, or ineligible run.

## V4 one-shot candidate

```powershell
& .\tvt_submission\run_all_local_after_gpu_free.ps1 `
  -Plan candidate-v4 `
  -Execute `
  -Acknowledgement "START_TVT_ONLY_WHEN_MACHINE_IS_IDLE" `
  -MinimumFreeGpuMiB 7000 `
  -MinimumFreeDiskGiB 20 `
  -Python "D:\Python\python.exe"
```

The wrapper refuses to overwrite an existing V4 output. Its result must not
be copied into the current formal paper. A successful diagnostic requires a
new prospective freeze before any later headline test.

## Full sequential queue

For completeness, `-Plan all` runs the fresh formal plan and then V4 while
holding the same global mutex. Separate idle windows are usually easier to
operate and audit.

At delivery time the GPU was occupied by pre-existing work and had only about
1.5 GiB of 8 GiB free. Therefore every execution form above must remain
blocked until the machine is genuinely idle. The default dry-run remains
safe. See `LOCAL_FORMAL_RUN_HANDOFF.md` for the artifact contract and
repository-root `HANDOFF.md` for next-agent state.
