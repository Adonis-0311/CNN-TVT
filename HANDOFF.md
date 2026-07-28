# VIMD-Net / IEEE TVT Workspace Handoff

Snapshot date: 2026-07-28

First GitHub push: 14:06 China Standard Time; the requested 13:55 deadline was
not met.

Repository: `https://github.com/Adonis-0311/CNN-TVT`

Status: **pre-submission engineering snapshot; not upload-ready**

## 1. Objective and non-negotiable boundaries

The project targets a strong IEEE Transactions on Vehicular Technology paper
on interference-robust automatic modulation classification (AMC). The patent
is the method-origin reference and the Idea document is the design guide.
This phase prepares the paper, evidence protocol, algorithms, tests, and local
execution handoff. It does **not** guarantee any acceptance probability.

Do not modify the two user source documents or the supplied IEEE archive:

- `D:\CNN信号调制识别\DFI257727-基于神经网络的干扰环境下信号调制识别方法及系统(定稿).docx`
  - SHA-256:
    `135403A345037FA8B71E8ED4AA094858EB37602ACC64A3F83BE8A672BE05141D`
- `D:\CNN信号调制识别\TVT_Flagship_VIMD_Net_AMC_Full_Design_Idea.md`
  - SHA-256:
    `E1DACF2B55C310E3D34AD4ECA27B77D8C570604B769AA35A989C86CD08DB5194`
- `C:\Users\Administrator\Downloads\IEEE-Transactions-LaTeX2e-templates-and-instructions.zip`
  - SHA-256:
    `6C315C3B6729BD7B96A6A0E7D3BB6342023413A4CD4D113FB4A193019AF1C603`

Do not touch, stop, resume, or reuse result artifacts from
`D:\CNN信号调制识别\tccn_satellite_amc`. Generic governance ideas were copied
into this project only after adaptation and provenance recording. Do not
touch unrelated work under `D:\Prepare`.

## 2. Running-process safety

At snapshot time, other local Python/GPU work was active, including a TCCN
Pilot-1 resume and an unrelated `D:\Prepare` experiment. This TVT task did not
start, stop, or signal them. Before any long TVT action, inspect current state:

```powershell
Get-CimInstance Win32_Process |
  Where-Object { $_.Name -match "python|matlab" } |
  Select-Object ProcessId,ParentProcessId,Name,CreationDate,CommandLine

nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version `
  --format=csv,noheader
```

If another training process is active or free VRAM is insufficient, do not
start TVT cache generation, training, candidate diagnostics, or latency
benchmarking. The local formal scripts are dry-run by default.

## 3. Authoritative project state

Project root: `D:\CNN信号调制识别\vimd_amc`

Implemented/frozen:

- VIMD A0--A7 family and mechanism probes;
- source-disjoint nine-domain factor-isolated protocol;
- exact-source paired views and hierarchical paired bootstrap;
- formal freeze: 47,000 source sequences, 94,000 paired views, 11 models,
  five seeds (`17,29,43,71,101`), 55 fits;
- MCLDNN, IQFormer-inspired, and 2025 CSSL-AMC
  official-architecture supervised adaptation comparator;
- CSSL source/commit/license lock; it is not a complete published-protocol
  reproduction and imports no official weights;
- VIMD-v4 DSBN candidate code, preregistered only and not executed;
- exact IEEE class copied from the user archive:
  `paper/IEEEtran.cls`, SHA-256
  `B0EB3567B81AEC7FE98144A3AD283EEAC2D31035BB19E0D9DCBA7DA190F18D9D`;
- paper internal-review evidence lock and author-verification gate;
- patent/Idea/paper traceability and 41-page patent render QA;
- deterministic pre-submission handoff packager (dry-run by default).

Current machine evidence:

- screening cache:
  `standards/cache_factor_screening_1024_v1`
  (excluded from Git and from paper claims);
- screening cache digest:
  `241b3aec6e74c79bac2d3ac22295098f0efe5cc79ff07acabf3593cbc32c49e3`;
- formal cache `standards/cache_factor_headline_1024_v1`: **absent**;
- formal run `artifacts/tvt_headline_1024_5seed_v1`: **absent**;
- `paper/release_lock.json`: **absent**;
- verified human author file: **absent**;
- formal training/cache generation launched by this task: **none**.

Therefore no diagnostic/screening number may be inserted into the paper and
no public submission PDF may be built.

## 4. Final integrated contract and validation

The non-training implementation and release chain are integrated and frozen.
No formal cache, training run, or result release was started. Final read-only
validation on 2026-07-28 produced:

- main project tests: **199/199 passed**;
- standards tests: **10/10 passed**;
- formal-freeze CLI: `valid=true`, `read_only=true`, 11 models, ordered seeds
  `17,29,43,71,101`, nine splits, and 47,000 declared sources;
- canonical internal PDF: `paper/build/main.pdf`, 8 pages, 420,895 bytes,
  SHA-256
  `E99D21895EF31FC8C4291777AE8F34D53842571CA649EE986E483409385DC636`;
- canonical paper source: `paper/main.tex`, 47,816 bytes, SHA-256
  `4485272A2D6EC55A313E6F2145B0A33A80BD7ED0D94137509FDA46A84E5A3F43`;
- internal paper gate: 36/36 checks passed, zero issues;
- release paper gate: deliberately fail-closed, 38/46 checks passed and eight
  expected failures because the formal release lock, verified authors,
  non-placeholder results, public review mode, and release sentinel are absent;
- all eight canonical PDF pages rendered and visually checked with no clipping,
  overlap, garbling, or out-of-bounds content.

Integrated contract:

- `metrics.py`, `evaluation.py`, and the runner preserve
  `target_profile_index` for clean-retention A/C/D versus B/E statistics.
- The formal reference is locked to the predeclared CSSL supervised adaptation;
  it must be described as the **primary reference**, never as post-hoc
  “strongest.” IQFormer-inspired remains a required non-oracle comparator.
- `generate_macro_values.py`, `validate_release.py`, `paper/main.tex`, and
  `paper/results_auto.tex` share a manifest-v3 artifact-derived contract:
  97 provenance records, 98 non-sentinel TeX commands after adding
  `ResultSource`, and 99 commands in a released file after adding
  `EligibleLockedResults`. The latency commands spell out the percentiles as
  `VIMDLatencyPFifty` and `VIMDLatencyPNinetyFive`; TeX command names containing
  digit tokens such as `P50` or `P95` are not part of the contract. Ordinary
  p50/p95 statistic labels remain valid in runner artifacts. The additional
  24 records expose the six missing A1/A2/A3/A4/A6/A7 hard macro-F1 means and
  the gain/family-wise simultaneous 95% interval for each predeclared A0--A7
  contrast.
- The formal run writes `ablation_paired_statistics.csv` for one six-contrast
  family. Its 33-column rows and source-aligned prediction NPZ bundles bind
  the lowercase 64-hex cache digest, exact ordered seeds
  `17,29,43,71,101`, class-stratified source clusters, source/SNR/SIR/profile
  alignment, bootstrap settings, and a finite strictly positive simultaneous
  critical value. The compact `run.json` summary binds the family and artifact;
  the generator and release validator deterministically rebuild the statistics
  instead of trusting that summary alone.
- The public paper must never render literal `generated`, `pending`, `TBD`,
  or manually typed performance cells.
- `validate_release.py` owns the single writer contract. Its validated security
  properties are:
  1. reconstruct the expected manifest from runner-native artifacts before
     writing;
  2. require canonical exact agreement, so a hand-edited
     `HardMacroFOneGain=+999.99` cannot unlock;
  3. parse only controlled comments and exactly the allowed one-line
     `\newcommand` set;
  4. reject unknown macros, `\input`, or other executable TeX;
  5. bind and re-render the exact result/table payload in release lock v2.
- `validate_paper_build.py` consumes the writer's single exported macro
  contract rather than inventing a second set. It verifies actual PDF page
  count, all transitive inputs (authors, references, figures, class, result
  files), final-log rerun warnings, release-lock v2, and exact Table IV
  consumption of all 24 A0--A7 extension macros.

The reviewer-attack and template-compliance documents under `docs/` retain
their observation-time findings (including earlier page counts and repair
items) as an audit trail. They are not the current release-status authority.
The final state is the validation record in this section together with
`tvt_submission/FINAL_CONVERGENCE_REPORT.md`.

## 5. Scientific release gates that must be executable

No eligible sentinel may be produced unless all administrative gates and all
of the following scientific gates pass:

1. A5 hard-interference macro-F1 is at least +5 percentage points versus
   **each** frozen non-oracle baseline: A0, MCLDNN, IQFormer-inspired, CSSL
   supervised adaptation.
2. At least two of `unseen_jammer`, `unseen_speed`, and `heldout_channel`
   achieve at least +3 pp macro-F1 versus the predeclared primary reference.
3. Clean retention is evaluated separately on seen A/C/D and held B/E.
   Each stratum requires point difference >= -1 pp and hierarchical paired
   95% CI lower bound >= -2 pp.
4. The six predeclared hard-region A0--A7 contrasts form one family:
   A3-A2 (teacher), A4-A3 (jammer/quality/orthogonality bundle), A5-A4
   (exact-source cross-condition contrast), A5-A1 (full versus single-mask),
   A5-A6 (composite full-versus-dual control), and A5-A7 (bounded additive
   bypass). Each family-wise simultaneous 95% macro-F1 lower bound must be
   strictly greater than zero both at full precision and after the public
   two-decimal percentage-point rendering; a displayed `+0.00` cannot unlock
   release.
5. Mechanism values must be finite and must not contradict any public
   mechanism claim. The visible metric is an oracle-conditioned spectral
   component ratio, not waveform SIR, SDR, or source separation.

The six contrasts share one class-stratified hierarchical paired bootstrap
over the exact five algorithm seeds and aligned test-source clusters. Its
family-wise interval is the non-studentized
`joint_max_absolute_centered_deviation_hierarchical_paired_bootstrap`, not a
bootstrap-t interval. A4-A3 is a bundled intervention and A5-A6 is a composite
control; neither supports a single-factor causal claim. This evidence uses the
already frozen A0--A7 fits and does not add to the 55-fit campaign.

Failure is a result, not permission to lower thresholds after test access.

## 6. Local execution entry points (do not run while other jobs are active)

Full formal handoff:
`tvt_submission/LOCAL_FORMAL_RUN_HANDOFF.md`

Complete idle-machine queue and safety interlock:
`tvt_submission/LOCAL_EXECUTION_QUEUE.md`

Dry preflight only:

```powershell
Set-Location -LiteralPath "D:\CNN信号调制识别\vimd_amc"
& .\tvt_submission\run_local.ps1 -Python "D:\Python\python.exe"
```

Formal cache, only when resources are free and a human explicitly starts it:

```powershell
& .\tvt_submission\run_local.ps1 `
  -Stage cache -Execute -Python "D:\Python\python.exe"
```

Formal 11-model/five-seed experiment, only after the formal cache passes:

```powershell
& .\tvt_submission\run_local.ps1 `
  -Stage experiment -Execute -Python "D:\Python\python.exe"
```

V4 is a separate candidate diagnostic and is not current paper evidence:

```powershell
& .\tvt_submission\run_candidate_local.ps1
# Add -Execute only under a separate approved, idle-GPU window.
```

The all-local wrapper is implemented, integration-tested, and dry-run by
default. Execution requires the exact acknowledgment documented in
`LOCAL_EXECUTION_QUEUE.md` and must refuse to start while foreign work or
insufficient free GPU memory is detected. No command in this handoff grants
permission to disrupt an active local task.

## 7. Immediate next-agent sequence

1. Read this handoff, `LOCAL_FORMAL_RUN_HANDOFF.md`, and the external patent
   and Idea sources; preserve all frozen hashes and contracts.
2. Confirm that no Python, MATLAB, LibreOffice, or foreign GPU workload is
   active; run the queue in dry-run mode first.
3. On an explicitly approved idle machine, build and validate the formal
   headline cache. Stop on any designation, checksum, component, or
   source-disjointness failure.
4. Run the frozen 11-model/five-seed campaign (55 fits) without changing the
   model family, reference, thresholds, seeds, or test protocol.
5. Generate the manifest from runner-native artifacts. Do not edit result
   macros or copy screening/diagnostic numbers.
6. Write the release only if every administrative and scientific gate passes;
   a negative result remains a valid outcome.
7. Human authors must then verify authorship, citations, claims, disclosure,
   and live TVT policy before a public build. The current internal PDF and
   handoff ZIP are explicitly not submission-ready.

## 8. Human-only blockers

The current TVT instructions state that AI tools may not replace an author to
generate article content and require acknowledgment disclosure when AI tools
modify author-written text. Human authors must substantively author, verify,
and disclose the final text; this engineering draft must not be uploaded
unchanged. Human authors must also verify every primary citation, authorship,
affiliations, funding/conflicts, patent-publication timing, and the policy in
force on the actual submission date.
