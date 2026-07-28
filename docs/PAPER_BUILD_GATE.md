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
`results_auto.tex`, but that file must still contain all 98 non-sentinel
macros exported by `validate_release.NON_SENTINEL_RESULT_MACROS` and no other
executable TeX content. The current release contract contains 97
provenance-bound macros and 99 macros including `ResultSource` and the release
sentinel. The build gate imports those tuples at runtime; it does not maintain
a second macro schema. A valid internal build is not submission evidence.

### Initial release

```powershell
& "D:\Python\python.exe" .\tvt_submission\validate_paper_build.py `
  --paper-root .\paper `
  --mode release
```

Release mode additionally requires:

- exactly one active `\internalreviewfalse`;
- `results_auto.tex` accepted byte-for-byte by the strict
  `validate_release.parse_results_auto` grammar, with all exported macros
  populated by non-placeholder values;
- every dynamically classified numeric result represented as one finite
  unit-free number, including all six additional A1/A2/A3/A4/A6/A7 hard
  macro-F1 means and all 18 members of the six ablation-contrast triples;
- every `Gain/CILow/CIHigh` triple having ordered bounds and containing its
  point gain, together with a positive integer parameter count and
  `VIMDLatencyPFifty <= VIMDLatencyPNinetyFive`;
- every exported public macro referenced in the source selected for a public
  build, so an internal-only mention cannot satisfy the contract; this
  explicitly binds the 24 new A0--A7 means/contrast leaves exactly once
  inside the `table` or `table*` environment that contains
  `\label{tab:ablations}`; moving one to decoy prose elsewhere in the public
  branch does not satisfy the table contract;
- `\EligibleLockedResults` defined as `eligible_locked_formal_run`, matching
  the evidence sentinel emitted by the result-release validator and required
  by `paper/main.tex`;
- a strict-JSON schema-v2 `paper/release_lock.json` with
  `submission_unlocked=true`, the eligible formal-run identity, a SHA-256
  binding to the exact `results_auto.tex`, and exact per-command provenance;
- no literal `pending`/`generated` residue in the selected public
  `\ifinternalreview` branches or as standalone extracted PDF cells; and
- a structurally parsed final PDF of at most 14 pages.

The release writer owns the scientific macro and provenance contract. The
paper gate consumes its exported macro groups, strict parser, and portable
lock validator instead of copying a second list. Its paper-side numeric check
only constrains the final TeX leaves to safe atomic numbers and verifies
ordering relationships that must remain true after serialization.

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
- a remaining rerun request for labels, cross-references, outlines,
  BibTeX, or Biber;
- no parseable final `Output written on ...` record;
- more than one ambiguous final-output record.

MiKTeX can wrap the word `pages` across physical log lines. The parser
normalizes whitespace for the final-output record and therefore accepts that
normal multi-line form. It reports the page count in JSON even in internal
mode.

The logged PDF byte count must exactly equal the audited PDF size. In
addition, a real `pdfinfo` parser must successfully read the selected PDF,
report the same file size, and report a positive page count exactly equal to
the log. Release page limits use this independently parsed page count, never
the log alone. Missing or failing `pdfinfo` is a fail-closed result.

Release mode also uses `pdftotext` as a defense-in-depth check for standalone
`pending` or `generated` table cells. A missing/failing extractor or an
unextractable public manuscript is rejected. Tool locations can be pinned with
`--pdfinfo` and `--pdftotext`; neither tool is allowed to write an output file.

## Freshness and multiple LaTeX passes

The gate audits the final log/PDF/recorder triplet. Multiple `latexmk` passes
are allowed; intermediate `.aux`, `.bbl`, and `.out` files under the selected
build directory are not treated as manuscript sources. The final log, PDF, and
`main.fls` must:

- be no more than 60 seconds apart by default;
- agree on exact PDF byte count; and
- bind the exact selected main source, result file, log, and PDF through
  recorder `INPUT`/`OUTPUT` entries.

Every actual recorder `INPUT` under the project tree is read and freshness
checked, including transitive TeX inputs, local classes, and figures.
Dependencies declared directly by `main.tex` are also checked; this explicitly
covers bibliography sources even when the final pdfLaTeX recorder lists only
the generated `.bbl`. In release mode the same rule covers the selected public
`authors_verified.tex` branch and `release_lock.json`. Therefore changing
prose, authors, citations, bibliography data, figures, local classes,
transitive inputs, result macros, or the release lock after the final build
forces a rebuild. A two-second filesystem timestamp tolerance is allowed by
default. Widening it weakens the provenance boundary and must not excuse a
known stale build.

## Paths and output contract

Defaults relative to `--paper-root` are:

| Role | Default |
|---|---|
| Main source | `main.tex` |
| Result macros | `results_auto.tex` |
| Final log | `build/main.log` |
| Final PDF | `build/main.pdf` |
| Recorder manifest | `build/main.fls` |
| Release lock | `release_lock.json` |

Paths can be selected with `--tex`, `--results`, `--log`, `--pdf`, `--fls`,
or `--release-lock`; relative paths resolve under `--paper-root`. The result
path is deliberately stricter: it must resolve to
`paper_root/results_auto.tex`, because `main.tex` is required to include that
exact file. Recorder bindings prevent an alternate source, log, or PDF from
being audited as a decoy.

Exit codes are:

- `0`: every check for the selected mode passed;
- `2`: an expected validation or argument failure;
- `3`: an unexpected validator failure, still reported as JSON.

Both successful and failed audits write their complete report to stdout. The
report includes artifact paths, sizes, timestamps, SHA-256 values, every
individual check, bounded log findings, page count, and failure reasons.

## Current internal-build observation

On 2026-07-28 the fresh
`paper/build_validation/main.{log,pdf,fls}` triplet passed all 36 applicable
internal checks. The PDF parsed independently as eight pages; the log reported
the same page and byte counts, no fatal error, unresolved citation/reference,
overfull box, or rerun request remained, every recorder binding was exact, and
all project inputs were fresh. The result is intentionally
`internal_build_validated=true` and `release_eligible=false`: the manuscript
still uses internal-review mode and safe placeholder result macros.

## Tests

```powershell
python -m unittest tests.test_paper_build_gate tests.test_public_table_contract -v
```

The synthetic fixtures cover:

- safe internal placeholders;
- a locked, non-placeholder 14-page release;
- fatal errors, undefined citations/references, overfull boxes, and rerun
  warnings;
- arbitrary MiKTeX line wrapping inside the final `pages` token;
- absent and unparseable logs;
- PDF/log byte disagreement and invalid PDF signatures;
- independently parsed PDF/log page disagreement, including a real 16-page
  PDF whose log falsely claims 14 pages;
- stale direct and transitive sources from `main.fls`, including authors,
  bibliography, figures, and local class files;
- a decoy `--results` override;
- public-source and extracted-PDF `pending`/`generated` residue;
- wrong review mode, placeholders, a missing lock, a missing evidence
  sentinel, an invalid lock digest, and page overflow;
- a black-box `generate_macro_values` -> `validate_release.write_release` ->
  paper-gate integration with the 97/98/99 exported macro counts, alphabetic
  `VIMDLatencyPFifty` / `VIMDLatencyPNinetyFive` names, scientific release
  gate, and provenance count checked;
- finite atomic numeric leaves plus invalid units, nonfinite values, and
  reversed confidence bounds;
- all six A0--A7 incremental/control means, all 18 ablation contrast values,
  point gains outside their confidence intervals, and public-versus-
  internal-only macro consumption;
- parsing the `table`/`table*` boundary containing `\label{tab:ablations}`,
  including rejection when any of its 24 new result macros is moved into
  decoy prose or appears more than once inside that table;
- rejection of digit-bearing TeX control-sequence names such as
  `VIMDLatencyP50`;
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
