# TVT paper-build static gate

`tvt_submission/validate_paper_build.py` is the final read-only check for an
already-built manuscript. It does not run LaTeX, update an auxiliary file,
touch a timestamp, or create a release artifact. It emits one JSON object on
stdout and exits nonzero whenever a required fact is absent, ambiguous, stale,
or inconsistent.

This gate complements `tvt_submission/validate_release.py`. The release
validator binds result values to formal run artifacts; the paper-build gate
checks that the PDF presented for review was actually rebuilt after the TeX,
result macros, and release lock reached their audited state.

## Modes

### Internal review

```powershell
& "D:\Python\python.exe" .\tvt_submission\validate_paper_build.py `
  --paper-root .\paper `
  --mode internal
```

Internal mode requires exactly one active line-level
`\internalreviewtrue` directive. It permits the safe placeholder
`results_auto.tex`, but that file must still contain the complete 74-command
public-table interface (3 text and 71 atomic numeric commands) and no other
executable TeX content. A valid internal build is not submission evidence.
The exact command and unit contract is recorded in
`docs/PUBLIC_TABLE_RELEASE_CONTRACT.md`.

### Initial release

```powershell
& "D:\Python\python.exe" .\tvt_submission\validate_paper_build.py `
  --paper-root .\paper `
  --mode release
```

Release mode additionally requires:

- exactly one active `\internalreviewfalse`;
- all result macros populated with non-placeholder values;
- every numeric result represented by one canonical, unit-free, finite scalar
  satisfying its table-specific bounds and confidence-interval ordering;
- `\EligibleLockedResults` defined as `eligible_locked_formal_run`, matching
  the evidence sentinel emitted by the result-release validator and required
  by `paper/main.tex`;
- a strict-JSON schema-v2 `paper/release_lock.json` with
  `submission_unlocked=true`, the eligible formal-run identity, a SHA-256
  binding to the exact `results_auto.tex`, and exact per-command provenance;
- a final PDF of at most 14 pages.

The 14-page default is the current regular initial-submission limit and
includes references and biographies. Revised/final regular papers currently
allow 16 pages; after a human rechecks the live TVT instructions, an
appropriate non-default threshold can be passed with `--max-pages 16`.

## LaTeX-log rejection rules

The final `paper/build/main.log` is rejected for any of:

- a TeX, LaTeX, package, or class fatal/error record;
- an undefined citation;
- an undefined reference;
- an overfull horizontal or vertical box;
- a remaining rerun request for labels, cross-references, BibTeX, or Biber;
- no parseable final `Output written on ...` record;
- more than one ambiguous final-output record.

MiKTeX can wrap the word `pages` across physical log lines. The parser
normalizes whitespace for the final-output record and therefore accepts that
normal multi-line form. It reports the page count in JSON even in internal
mode.

The logged PDF byte count must exactly equal the audited PDF size, and the
file must start with a PDF signature. This binds the selected PDF to the final
record in the selected log without adding a PDF library dependency.

## Freshness and multiple LaTeX passes

The gate audits only the final log/PDF pair. Multiple `latexmk` passes are
allowed; intermediate `.aux`, `.bbl`, and `.out` timestamps are intentionally
ignored. The final log and PDF must:

- be no more than 60 seconds apart by default;
- agree on exact PDF byte count; and
- be at least as new as `main.tex` and `results_auto.tex`, allowing a
  two-second filesystem timestamp tolerance.

Release mode applies the same freshness rule to `release_lock.json`.
Therefore, changing prose, citations, result macros, or the release lock after
the final build makes the previous PDF stale and forces a rebuild. The
tolerances can be changed explicitly for an unusual filesystem, but widening
them weakens the provenance boundary and must not be used to excuse a known
stale build.

## Paths and output contract

Defaults relative to `--paper-root` are:

| Role | Default |
|---|---|
| Main source | `main.tex` |
| Result macros | `results_auto.tex` |
| Final log | `build/main.log` |
| Final PDF | `build/main.pdf` |
| Release lock | `release_lock.json` |

Each can be overridden with `--tex`, `--results`, `--log`, `--pdf`, or
`--release-lock`. Relative overrides are resolved under `--paper-root`.

Exit codes are:

- `0`: every check for the selected mode passed;
- `2`: an expected validation or argument failure;
- `3`: an unexpected validator failure, still reported as JSON.

Both successful and failed audits write their complete report to stdout. The
report includes artifact paths, sizes, timestamps, SHA-256 values, every
individual check, bounded log findings, page count, and failure reasons.

## Current internal-build observation

On 2026-07-28 the checked-in log and PDF parsed cleanly as a seven-page build:
there were no fatal errors, undefined citations/references, overfull boxes, or
rerun requests, and the logged byte count matched the PDF. The static gate
nevertheless rejected that build because `paper/main.tex` had been updated
after the PDF/log pair. This is the intended fail-closed outcome; a fresh
internal compile is required before the same command can return zero.

## Tests

```powershell
python -m unittest tests.test_paper_build_gate -v
```

The synthetic fixtures cover:

- safe internal placeholders;
- a locked, non-placeholder 14-page release;
- fatal errors, undefined citations/references, overfull boxes, and rerun
  warnings;
- absent and unparseable logs;
- PDF/log byte disagreement and invalid PDF signatures;
- stale sources;
- wrong review mode, placeholders, a missing lock, a missing evidence
  sentinel, an invalid lock digest, and page overflow;
- JSON stdout plus nonzero CLI failure behavior; and
- a read-only audit of the checked-in internal build.

The tests use temporary files and never compile the paper or launch an
experiment.

## Checks that remain human-owned

This static gate does not prove every TVT formatting rule. Human final review
must still confirm the current IEEE two-column 10-point transaction format,
single-column title treatment, inclusion of references/biographies in the
page count, the current instruction to remove international-language fonts,
live submission policies, and visual legibility of every rendered page.
