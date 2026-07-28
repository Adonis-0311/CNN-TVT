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

## 4. Urgent WIP contract at this snapshot

This GitHub upload is intentionally a checkpoint while the final release-chain
repair is still in progress. Python syntax was checked immediately before the
snapshot, but the full test suite has not yet been rerun after the latest
parallel edits.

Known integration state:

- `metrics.py`, `evaluation.py`, and runner changes are carrying
  `target_profile_index` toward clean-retention A/C/D versus B/E statistics.
- The formal reference is locked to the predeclared CSSL supervised adaptation;
  it must be described as the **primary reference**, never as post-hoc
  “strongest.” IQFormer-inspired remains a required non-oracle comparator.
- `generate_macro_values.py`, `validate_release.py`, `paper/main.tex`, and
  `paper/results_auto.tex` share a manifest-v3 artifact-derived contract:
  73 provenance records, 74 non-sentinel TeX commands after adding
  `ResultSource`, and 75 commands in a released file after adding
  `EligibleLockedResults`. The latency commands spell out the percentiles as
  `VIMDLatencyPFifty` and `VIMDLatencyPNinetyFive`; TeX command names containing
  digit tokens such as `P50` or `P95` are not part of the contract. At snapshot
  time, final integration and full regression remain pending.
- The public paper must never render literal `generated`, `pending`, `TBD`,
  or manually typed performance cells.
- `validate_release.py` owns the single writer contract. Preserve/finish these
  security properties:
  1. reconstruct the expected manifest from runner-native artifacts before
     writing;
  2. require canonical exact agreement, so a hand-edited
     `HardMacroFOneGain=+999.99` cannot unlock;
  3. parse only controlled comments and exactly the allowed one-line
     `\newcommand` set;
  4. reject unknown macros, `\input`, or other executable TeX;
  5. bind and re-render the exact result/table payload in release lock v2.
- `validate_paper_build.py` must consume the writer's single exported macro
  contract rather than invent a second set. It must verify actual PDF page
  count, all transitive inputs (authors, references, figures, class, result
  files), final-log rerun warnings, and release-lock v2.

The independent reviewer audit is:
`docs/FINAL_TVT_REVIEWER_ATTACK_AUDIT.md`.

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
4. A5 must improve over the single-mask A1 and dual-route A6 controls in the
   hard region if the paper retains route-count/tri-mask improvement language.
5. Mechanism values must be finite and must not contradict any public
   mechanism claim. The visible metric is an oracle-conditioned spectral
   component ratio, not waveform SIR, SDR, or source separation.

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

The all-local wrapper is implemented and dry-run by default; its final
integration audit remains part of the pending regression. Execution requires
the exact acknowledgment documented in `LOCAL_EXECUTION_QUEUE.md` and must
refuse to start while foreign work or insufficient free GPU memory is detected.
No command in this handoff grants permission to disrupt an active local task.

## 7. Immediate next-agent sequence

1. Read this file, `TVT_Flagship...Idea.md` (external), the patent traceability
   report, and `docs/FINAL_TVT_REVIEWER_ATTACK_AUDIT.md`.
2. Inspect current diffs; do not roll back parallel/user changes.
3. Make `validate_release.py` the only source of the macro/table contract.
4. Reconcile writer, generator, paper placeholders, and paper gate.
5. Complete clean profile strata and scientific promotion tests.
6. Run targeted tests, then the entire main and standards test suites.
7. Run formal-freeze validation and dry-run scripts; do not start training.
8. Recompile the internal paper with local `IEEEtran.cls`, render all pages,
   and perform visual QA.
9. Update evidence/readiness/handoff documents with exact test counts and
   checksums.
10. Build the explicitly non-upload-ready author handoff archive. Public
    release remains locked until formal results and human authorship exist.

## 8. Human-only blockers

The current TVT instructions state that AI tools may not replace an author to
generate article content and require acknowledgment disclosure when AI tools
modify author-written text. Human authors must substantively author, verify,
and disclose the final text; this engineering draft must not be uploaded
unchanged. Human authors must also verify every primary citation, authorship,
affiliations, funding/conflicts, patent-publication timing, and the policy in
force on the actual submission date.
