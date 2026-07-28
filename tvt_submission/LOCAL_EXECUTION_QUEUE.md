# TVT Local Simulation and Training Queue

Status date: 2026-07-28

This file hands every planned local long-running entry point to the operator.
Nothing described here was started by the delivery turn. The wrapper
`run_all_local_after_gpu_free.ps1` is dry-run by default, never waits for or
terminates another process, and refuses execution when foreign Python,
MATLAB, or LibreOffice work is visible or when free GPU memory is below the
configured floor.

## Included plans

| Plan | Content | Paper evidence |
|---|---|---|
| `preflight` | Validate the formal freeze and print both formal and V4 commands | No |
| `formal` | Build the formal MATLAB TDL cache; run 11 models × 5 seeds; generate artifact-derived macros/tables; validate release | Only if every gate passes |
| `candidate-v4` | Run the preregistered one-shot VIMD-v4 DSBN diagnostic on the screening cache | No |
| `all` | Run `formal`, then the separate V4 diagnostic | Formal part only; V4 stays non-evidence |

The formal plan is sequential. It does not launch 55 concurrent GPU jobs.
The V4 plan is deliberately separate from the frozen formal family.

## Safe dry-run now

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

## Formal run after the machine becomes idle

Do not run this while TCCN, another Python experiment, MATLAB, LibreOffice, or
another GPU job is active:

```powershell
& .\tvt_submission\run_all_local_after_gpu_free.ps1 `
  -Plan formal `
  -Execute `
  -Acknowledgement "START_TVT_ONLY_WHEN_MACHINE_IS_IDLE" `
  -MinimumFreeGpuMiB 7000 `
  -Python "D:\Python\python.exe"
```

This performs release preflight but does not write
`paper/results_auto.tex` or `paper/release_lock.json`. If and only if the
formal run is eligible, every scientific promotion gate passes, and the human
authors approve release, repeat the command with `-WriteRelease`.

## V4 one-shot candidate after the machine becomes idle

```powershell
& .\tvt_submission\run_all_local_after_gpu_free.ps1 `
  -Plan candidate-v4 `
  -Execute `
  -Acknowledgement "START_TVT_ONLY_WHEN_MACHINE_IS_IDLE" `
  -MinimumFreeGpuMiB 7000 `
  -Python "D:\Python\python.exe"
```

The wrapper refuses to overwrite an existing V4 output. Its result must never
be copied into the current formal paper. A successful candidate requires a
new prospective freeze before any later headline test.

## Full sequential queue

The following is provided for completeness but is usually less convenient
than running the formal and candidate plans in separate idle windows:

```powershell
& .\tvt_submission\run_all_local_after_gpu_free.ps1 `
  -Plan all `
  -Execute `
  -Acknowledgement "START_TVT_ONLY_WHEN_MACHINE_IS_IDLE" `
  -MinimumFreeGpuMiB 7000 `
  -Python "D:\Python\python.exe"
```

## Current-machine result

At delivery time the GPU was occupied by pre-existing work and only about
1.5 GiB of 8 GiB was free. The execution form above therefore must fail
closed. This is intentional and proves that the handoff cannot disrupt the
currently running local jobs. The dry-run form remains safe.

For the complete formal artifact contract, stop conditions, expected paths,
and public-build sequence, read `LOCAL_FORMAL_RUN_HANDOFF.md`. For the
next-agent state and known WIP integration points, read the repository-root
`HANDOFF.md`.

