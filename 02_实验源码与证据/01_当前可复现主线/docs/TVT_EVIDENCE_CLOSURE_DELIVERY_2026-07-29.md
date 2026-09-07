# TVT v2 evidence-closure delivery

Date: 2026-07-29
Repository: `D:\CNN信号调制识别\TVT支线\02_实验源码与证据\01_当前可复现主线`
Delivery state: **simulation-only initial-manuscript evidence chain closed to
the pre-compute boundary; formal v2 numerical evidence not yet executed**

This is the controlling status record for the TVT v2 initial manuscript.
Within its declared source-disjoint TDL-profile simulation scope, everything
that can be delivered without consuming the local MATLAB/CUDA formal campaign
is implemented or documented: the frozen protocol and prospective statistical
justification, cache/training runners, scientific validation, source-bound
paper-macro promotion, release lock, manuscript wiring, tests, internal
placeholder source, and operator handoff.

The only remaining evidence-generating action for this initial draft is the
single local v2 queue. If it exits zero, the remaining manuscript work is
deterministic artifact-derived macro promotion, evidence-conditioned prose
selection, and an internal rebuild. No formal v2 performance value is claimed
in this delivery. Authorship, journal-policy, cover-letter, upload, and other
submission operations are intentionally outside this closure decision.

## 1. Current closure boundary

| Area | State | Meaning |
|---|---|---|
| Static implementation | **Delivered** | Method, teacher forms, receiver stress, frozen protocol, non-overridable CUDA/device contract, executable-source closure, runners, gates, tests, and manuscript wiring are present. |
| Local formal computation | **Not executed** | 10k/30k/100k caches, learning curve, 120 fits, predictions, intervals, and scientific gate await one local command. |
| Initial-draft backfill automation | **Delivered, not invoked** | A passing hash-bound v2 run can deterministically generate and validate every paper macro; no manually typed result is admitted. |
| Internal manuscript source | **Delivered** | The initial manuscript is wired to placeholders and remains fail-closed until eligible values are promoted. |
| RadioML/MFENet/flagship expansion | **Outside current claim scope** | No native-public-benchmark, MFENet reproduction, SDR/field, full-V2X, cochannel, or multi-jammer result is claimed; adding one requires a separate prospective evidence path. |
| Submission workflow | **Not assessed here** | Authorship, policy, cover letter, public release, archive, and upload are deliberately deferred and do not reopen the scientific initial-draft chain. |

Historical v1 caches, runs, five-seed protocols, macro manifests, release
scripts, and status documents are provenance only. They may not be
reinterpreted, merged, or promoted as v2 evidence.

## 2. Delivered scientific and protocol implementation

### 2.1 Teacher and mechanism

- The primary fixed component-power teacher is
  \(M_s=[q_s-q_j]_+\), \(M_j=[q_j-q_s]_+\), and
  \(M_o=q_u+2\min(q_s,q_j)\).
- A3′ (`a3p_tri_proportional_teacher`) supplies the frozen proportional
  teacher control; A3−A3′ is confirmatory.
- The local-whitened teacher remains diagnostic-only.
- Hard-split teacher audits report mean, P90, and nonzero fraction by family
  and exact SNR/SIR cell.
- The six-family periodic-Hann occupancy/gain diagnostic is preregistered and
  report-mandatory. Its direction does not automatically gate an honest null
  submission, but it gates a positive mechanism claim.

Anchors:

- `src/vimd_amc/models/spectral.py`
- `src/vimd_amc/ablation.py`
- `src/vimd_amc/teacher_audit.py`
- `experiments/audit_hard_split_teacher.py`
- `experiments/evaluate_occupancy_gain_mechanism.py`
- `docs/TEACHER_FUNCTION_EVIDENCE.md`

### 2.2 Frozen v2 campaign

- Freeze: `tvt_submission/configs/formal_tvt_freeze_v2.json`.
  SHA-256:
  `926e8eebdfadaab7876686f4762ec9bf2ffbb564a3e013b46932598f0a25b423`.
- Prospective statistical justification:
  `docs/TVT_V2_PROSPECTIVE_STATISTICAL_JUSTIFICATION.md`.
  SHA-256:
  `055b330c20ef7a908ef191b75bb7a39302c1c51be29a1d3b517639002cae26a7`.
- Cache master seed: `20260727`; bootstrap seed: `20260803`.
- Learning curve: 10k/30k/100k, A0/A5, seeds
  `17,29,43,71,101`, with per-epoch train/validation macro-F1 and loss.
- The learning curve cannot select the final scale or retune
  hyperparameters; the formal scale stays 100k.
- Formal grid: 12 frozen models × seeds
  `17,29,43,71,101,131,173,211,257,307` = 120 sequential fits.
- Confirmatory family:

  1. A5−A0;
  2. A3−A3′; and
  3. A5−A6.

- All other ablation contrasts are exploratory.
- Historical v1 results are explicitly ineligible for v2 reinterpretation.
- The frozen `device=cuda` value is not a convenience default: learning and
  formal preflight reject `cpu`, `auto`, `cuda:0`, or any other CLI override.
- The `tvt_v2_execution` source closure covers the complete
  `src/vimd_amc` tree, both v2 runners, the standard runner, freeze/contract,
  unified PowerShell wrapper, cache builder, MATLAB backend, and v2
  validators. It is rechecked after cache generation, after learning-curve
  execution, at each standard-run boundary, and at the scientific gate.
- Run evidence binds the Python, PyTorch, CUDA, NumPy, Pandas, and SciPy
  versions plus the GPU identity; environment evidence therefore accompanies,
  rather than substitutes for, the executable digest.

Anchors:

- `tvt_submission/formal_v2_contract.py`
- `experiments/run_learning_curve_v2.py`
- `experiments/run_formal_tvt_v2.py`
- `tvt_submission/validate_formal_freeze_v2.py`

### 2.3 Fail-closed scientific release

`tvt_submission/validate_v2_release.py` rejects incomplete or inconsistent
evidence, including:

- any missing/extra model-seed fit;
- fallback or ineligible checkpoints;
- missing or mismatched result/checkpoint/history artifacts;
- any missing model/seed/regime prediction NPZ;
- cache, split, source-ID, label, profile, SNR, or SIR mismatch;
- a confirmatory CSV that does not exactly rederive from frozen prediction
  bundles and the simultaneous-bootstrap contract;
- incomplete/failing predeclared OOD, ADC, synchronization, or
  clean-retention branches; and
- missing required occupancy/gain reporting.
- a device override or mismatch with the frozen CUDA execution contract;
- drift in any member of the full v2 executable-source closure; and
- incomplete or inconsistent runtime/library/GPU environment evidence.

A passing gate means `scientific_evidence_passed=true`. It intentionally keeps
`submission_unlocked=false` until the separate macro/release-lock stage and
human review have passed.

### 2.4 Canonical v2 paper promotion

`tvt_submission/validate_v2_paper_release.py` is the single v2 bridge from
immutable runner artifacts to manuscript values. It:

- accepts only the exact frozen run, learning evidence, scientific gate, cache
  bindings, and source hashes;
- rederives manuscript values from JSON/CSV/NPZ artifacts;
- covers headline, regime, clean/stress, A3′, confirmatory/exploratory,
  receiver, occupancy, complexity, and latency macros;
- writes an auditable macro manifest without manual performance arguments;
- atomically binds `paper/results_auto.tex` and `paper/release_lock.json`;
- rejects placeholders, unexpected/missing macros, mutation, duplicate
  evidence, nonfinite values, and digest mismatch; and
- is selected automatically by `validate_paper_build.py` for the v2
  manuscript contract.

The local training queue generates
`tvt_submission/formal_macro_values_v2.json` only after the scientific gate
passes. It deliberately does not modify the manuscript or release lock.

## 3. Internal-review manuscript

Validated artifact:

`paper/build_initial_v2/main.pdf`

Current placeholder-build SHA-256:

`d683cb2317279d54236a8e5cbc4ea417c9405d96996aff6b10824000d04e2c32`

Current size and validation:

- 438,006 bytes;
- 9 pages;
- no fatal LaTeX errors;
- no undefined citations or references;
- no overfull boxes;
- no rerun requirement; and
- PDF/log/FLS/source dependency checks passed.

This PDF contains controlled result placeholders and is **not** a numerical
evidence artifact. It is a fresh rendering of the current initial-draft source
and passed the internal build gate; any later source edit requires another
rebuild before review.

## 4. Only remaining machine-compute action

The safe dry preflight starts no work:

```powershell
Set-Location -LiteralPath "D:\CNN信号调制识别\TVT支线\02_实验源码与证据\01_当前可复现主线"
& .\tvt_submission\run_v2_after_gpu_free.ps1 `
  -Python "D:\Python\python.exe"
```

When the human operator confirms the machine is genuinely idle:

```powershell
Set-Location -LiteralPath "D:\CNN信号调制识别\TVT支线\02_实验源码与证据\01_当前可复现主线"
& .\tvt_submission\run_v2_after_gpu_free.ps1 `
  -Execute `
  -Acknowledgement "START_TVT_ONLY_WHEN_MACHINE_IS_IDLE" `
  -MinimumFreeGpuMiB 7000 `
  -MinimumFreeDiskGiB 40 `
  -Python "D:\Python\python.exe"
```

This is the only supported fresh compute entry point. The wrapper owns
`Global\VIMD_AMC_TVT_LOCAL_EXECUTION_V2`, refuses to preempt active
Python/MATLAB work, builds and validates all three caches, runs the frozen
learning curve and 120-fit formal grid, derives the scientific gate, and
generates the source-bound macro manifest.

At delivery time the required formal directories/files are intentionally
absent:

- `standards/cache_factor_learning_10k_1024_v2`
- `standards/cache_factor_learning_30k_1024_v2`
- `standards/cache_factor_headline_1024_v2`
- `artifacts/tvt_learning_curve_v2/learning_curve_evidence.json`
- `artifacts/tvt_headline_1024_10seed_v2/run.json`
- `artifacts/tvt_headline_1024_10seed_v2/v2_scientific_release_gate.json`
- `tvt_submission/formal_macro_values_v2.json`

Consequently, learning saturation, formal macro-F1, simultaneous intervals,
calibrated OOD/receiver/clean outcomes, and the formal occupancy direction are
unknown. No direction may be asserted before these artifacts exist and pass.

The detailed operator contract, output inventory, and validated cache reuse
syntax are in `tvt_submission/LOCAL_EXECUTION_QUEUE.md`.

## 5. Post-compute initial-draft backfill

After the queue exits zero:

1. Run the read-only `--validate-macro-manifest` action shown in
   `tvt_submission/LOCAL_EXECUTION_QUEUE.md`.
2. Inspect every scientific branch and select only the prewritten
   evidence-conditioned wording supported by its gate; a failed or null branch
   narrows the corresponding statement rather than triggering retuning.
3. Explicitly invoke `validate_v2_paper_release.py --write-release` as the
   atomic evidence-lock/backfill operation. This does not authorize upload.
4. Run `validate_v2_paper_release.py --validate-release` against the written
   macros, lock, and still-present source artifacts.
5. Rebuild the internal manuscript and inspect every populated table,
   confidence interval, gate-dependent statement, and limitation.

The evidence chain ends at this populated, internally reviewed initial draft.
Authorship metadata, public-branch compilation, journal policy, source archive,
and upload remain a later submission workflow and are not conditions for this
closure state. No automatic step may infer submission approval from a zero
exit code.

## 6. Scope-excluded extensions

The following are not hidden gaps in the current simulation-only claim. They
are optional claim expansions that require their own prospective evidence:

- **RadioML/public-data transfer:** if an authorized dataset is acquired,
  record immutable hashes,
  license/provenance, native preprocessing/split checks, and separate clean
  and prospectively frozen interference-overlay results. The current initial
  draft already makes no public-benchmark transfer claim.
- **MFENet:** use only an authorized, hash-pinned author implementation or a
  dated human contact trail before adding a numerical row or superiority
  claim. The current draft cites MFENet only as related work.
- **Flagship expansion:** SDR/field transfer, complete V2X geometry,
  dominant-emitter cochannel, mixed/multi-jammer, native public datasets, and
  broader device/latency evidence are excluded by the current Scope paragraph.

The package does not fabricate external artifacts, send unauthorized
correspondence, or imply that an excluded extension was measured.

## 7. Acceptance checklist

Formal-run and deterministic-backfill evidence:

- [ ] Three v2 cache manifests exist, validate, and are hash-pinned.
- [ ] Learning evidence is complete and exactly bound to all scale runs.
- [ ] All 120 formal fits have eligible checkpoints with no fallback.
- [ ] Every required prediction bundle is present and cache/source-bound.
- [ ] The rederived simultaneous confirmatory family agrees exactly.
- [ ] Every OOD, receiver, clean, and occupancy branch is reported.
- [ ] `v2_scientific_release_gate.json` passes.
- [ ] `formal_macro_values_v2.json` is generated from that exact run.
- [ ] Explicit `--write-release` succeeds without replacement shortcuts.
- [ ] Explicit `--validate-release` passes against canonical lock bytes and
  unchanged source artifacts.
- [ ] The populated internal draft is rebuilt and every result/limitation is
  checked against the locked artifact.

Submission administration and optional scope expansions are deliberately not
part of this checklist.

## 8. Prohibited shortcuts

- Do not promote screening, diagnostic, smoke, oracle, historical-v1, or
  source-mutated results.
- Do not type performance values into TeX or the macro manifest.
- Do not alter the freeze after viewing results.
- Do not reuse a partial cache without exact manifest pins and validation.
- Do not start below the idle/GPU/disk gates or terminate unrelated work.
- Do not equate `scientific_evidence_passed` with publication approval.
- Do not write a release PDF before the v2 macro manifest and release lock
  pass.
- Do not claim RadioML/MFENet evidence that does not exist.

## 9. Controlling files

- This closure record:
  `docs/TVT_EVIDENCE_CLOSURE_DELIVERY_2026-07-29.md`
- Operator queue: `tvt_submission/LOCAL_EXECUTION_QUEUE.md`
- Package overview: `tvt_submission/README.md`
- Reviewer matrix: `docs/TVT_REVIEW_RESPONSE_MATRIX.md`
- Teacher evidence: `docs/TEACHER_FUNCTION_EVIDENCE.md`
- v2 protocol: `docs/TVT_V2_M4_M7_PROTOCOL.md`
- v2 freeze: `tvt_submission/configs/formal_tvt_freeze_v2.json`
- v2 execution: `tvt_submission/run_v2_after_gpu_free.ps1`
- scientific gate: `tvt_submission/validate_v2_release.py`
- paper bridge: `tvt_submission/validate_v2_paper_release.py`
- build gate: `tvt_submission/validate_paper_build.py`
- internal manuscript: `paper/main.tex`
- internal PDF: `paper/build_initial_v2/main.pdf`

## 10. Bottom line

Within the declared simulation-only initial-manuscript scope, the static
evidence chain is closed and formal numerical claims remain honestly blank.
The only missing scientific evidence is produced by the single local v2
queue; after a zero exit, artifact-derived macro promotion and an internal
rebuild complete the initial draft. RadioML, MFENet reproduction, flagship
hardware extensions, and submission administration are not silently pending
requirements for this scoped claim.
