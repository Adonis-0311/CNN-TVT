# TVT v2 convergence package

Status date: 2026-07-29

This directory is the fail-closed handoff for the simulation-only TVT v2
initial manuscript. The implementation, prospective statistical rationale,
frozen protocol, scientific validator, paper-macro promotion bridge, release
validator, tests, and internal placeholder manuscript are ready. The only
remaining evidence-generating step is the preregistered local MATLAB/CUDA
execution; a successful run is followed by deterministic macro backfill and
an internal rebuild.

No formal v2 numerical result exists at delivery time. Historical v1 files in
this directory remain available for provenance only and must not be used as a
shortcut or reinterpreted as v2 evidence.

## One local-compute entry point

Run every command from the repository root. A command without `-Execute` is a
safe dry preflight:

```powershell
Set-Location -LiteralPath "D:\CNN信号调制识别\TVT支线\02_实验源码与证据\01_当前可复现主线"
& .\tvt_submission\run_v2_after_gpu_free.ps1 `
  -Python "D:\Python\python.exe"
```

When the machine is genuinely idle, the human operator starts the complete
fresh queue with exactly:

```powershell
Set-Location -LiteralPath "D:\CNN信号调制识别\TVT支线\02_实验源码与证据\01_当前可复现主线"
& .\tvt_submission\run_v2_after_gpu_free.ps1 `
  -Execute `
  -Acknowledgement "START_TVT_ONLY_WHEN_MACHINE_IS_IDLE" `
  -MinimumFreeGpuMiB 7000 `
  -MinimumFreeDiskGiB 40 `
  -Python "D:\Python\python.exe"
```

Do not invoke lower-level cache or training runners opportunistically. The
wrapper owns a global mutex, refuses to preempt active Python/MATLAB work,
checks GPU and disk capacity, builds and validates the three v2 caches, runs
the 10k/30k/100k learning curve, executes the frozen 12-model × 10-seed grid,
derives the scientific gate, and writes the artifact-derived v2 macro
manifest. It does **not** overwrite `paper/results_auto.tex`, create
`paper/release_lock.json`, or unlock a submission.

Exact operator instructions, expected artifacts, continuation rules, and the
post-run release commands are in
[`LOCAL_EXECUTION_QUEUE.md`](LOCAL_EXECUTION_QUEUE.md).

## Frozen evidence contract

- Freeze: `configs/formal_tvt_freeze_v2.json`.
- Formal cache: `standards/cache_factor_headline_1024_v2`, with 100,000
  training sources and the frozen validation/test/stress splits.
- Learning curve: A0/A5 at 10k, 30k, and 100k, five fixed seeds. It documents
  scale adequacy and cannot select the formal scale or retune the model.
- Formal run: `artifacts/tvt_headline_1024_10seed_v2`, 12 models and seeds
  `17,29,43,71,101,131,173,211,257,307`, for 120 sequential fits.
- Confirmatory family: A5−A0, A3−A3′, and A5−A6 under one simultaneous
  hierarchical paired-bootstrap family.
- Scientific validator: `validate_v2_release.py`.
- Paper promotion/release validator: `validate_v2_paper_release.py`.
- Strict paper-build validator: `validate_paper_build.py`, which selects the
  v2 macro contract when the v2 manuscript is present.

The CSSL-AMC comparator is an official-architecture supervised adaptation, not
a full reproduction or an official reported result. IQFormer-inspired and the
MCLDNN reimplementation retain their bounded provenance labels. Screening,
diagnostic, oracle-control, source-mutated, historical-v1, and manually typed
performance values are ineligible.

## Machine outputs expected after a successful queue

- `standards/cache_factor_learning_10k_1024_v2/manifest.json`
- `standards/cache_factor_learning_30k_1024_v2/manifest.json`
- `standards/cache_factor_headline_1024_v2/manifest.json`
- `artifacts/tvt_learning_curve_v2/learning_curve_evidence.json`
- `artifacts/tvt_headline_1024_10seed_v2/run.json`
- `artifacts/tvt_headline_1024_10seed_v2/v2_scientific_release_gate.json`
- `tvt_submission/formal_macro_values_v2.json`

The queue is successful only when it exits zero. Existence alone does not make
an artifact eligible: every hash, cache/source binding, fit/checkpoint
membership, prediction bundle, simultaneous interval, OOD/receiver/clean
branch, and required mechanism report must pass its validator.

## Initial-manuscript boundary

After the local queue succeeds, every scientific branch must be reviewed and
only the evidence-conditioned wording supported by its gate may remain. The
explicit `--write-release` operation documented in
`LOCAL_EXECUTION_QUEUE.md` then acts as the atomic evidence-lock/backfill step:
it writes the paper macros and release lock from the exact artifacts. The
populated internal manuscript must be rebuilt and inspected. This operation
does not itself authorize submission.

The current draft already adopts the narrower claim: it does not claim native
RadioML transfer, MFENet reproduction/superiority, SDR/field transfer,
complete V2X geometry, or cochannel/multi-jammer performance. Those are
optional future scope expansions, not unresolved inputs to the present
evidence chain. Authorship, journal policy, cover letter, public source
archive, and upload are a later submission workflow and are not assessed here.

The controlling closure record is
[`../docs/TVT_EVIDENCE_CLOSURE_DELIVERY_2026-07-29.md`](../docs/TVT_EVIDENCE_CLOSURE_DELIVERY_2026-07-29.md).
